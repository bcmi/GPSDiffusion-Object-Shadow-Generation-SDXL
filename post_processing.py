import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything
from PIL import Image
import os
from train_post_process_predictor import PostProcess
from tqdm import tqdm
import argparse

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

@torch.no_grad()
def restore_img(comp_img):
    comp_img = comp_img * 255
    return comp_img

def get_args_parser():
    parser = argparse.ArgumentParser('Post-processing', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--checkpoint_path', default='./models/pretrained_models/Shadow_ppp.ckpt', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--result_dir_xl', default='./results_sdxl', type=str)
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    model = PostProcess(infe_steps=50).cpu()
    model.load_state_dict(load_state_dict(args.checkpoint_path, location='cuda'), strict=False)
    model = model.cuda()

    # seed = 723

    ddim_steps = 50
    
    fg_instance_path = os.path.join(args.result_dir_xl, "gt_object_mask")
    shadowfree_path = os.path.join(args.result_dir_xl, "gt_shadowfree_img")
    generated_img_root = os.path.join(args.result_dir_xl, "gen_result")

    save_root = os.path.join(args.result_dir_xl, "pp_result")  
    save_mask_root = os.path.join(args.result_dir_xl, "pp_mask")

    imname_total= os.listdir(generated_img_root)

     #seed_everything(seed)
    for img_name in tqdm(imname_total):
        shadowfree_img_path = os.path.join(shadowfree_path, img_name)
        object_mask_path = os.path.join(fg_instance_path, img_name)
        
        width, height = 256, 256
        shadowfree_img = cv2.imread(shadowfree_img_path)
        shadowfree_img = cv2.resize(shadowfree_img, (width, height))
        object_mask = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
        object_mask = cv2.resize(object_mask, (width, height))

        
        shadowfree_img = cv2.cvtColor(shadowfree_img, cv2.COLOR_BGR2RGB)

        source = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)

        source = source.astype(np.float32) / 255.0
        
        shadowfree_img_ = (shadowfree_img.astype(np.float32) / 127.5) - 1.0

        device = torch.device("cuda:"+str(args.gpu_id))
        shadowfree_img_ = torch.from_numpy(shadowfree_img_.copy()).float().unsqueeze(0).to(device)
        source_ = torch.from_numpy(source.copy()).float().unsqueeze(0).to(device)

        prompt = ''
        batch = dict(txt=[prompt], hint=source_, mask=None, name=img_name)

        comp_img_scaled = batch['hint'][:, :, :, :3]
        obj_mask = batch['hint'][:, :, :, 3:]
        comp_img= restore_img(comp_img_scaled)

        with torch.no_grad():
            model.eval()
            image_scaled = np.array(Image.open(os.path.join(generated_img_root,img_name)).convert('RGB').resize((width, height),Image.NEAREST))
            image_scaled = torch.from_numpy(image_scaled.copy()).float().unsqueeze(0).to(device)
            image_scaled = image_scaled/127.5 - 1
            image = torch.clamp(image_scaled, -1., 1.)
            image = (image + 1.0) / 2.0
            image_256 = (image * 255).int()

        input = torch.concat([image_scaled, comp_img_scaled * 2 - 1, obj_mask], dim=-1)
        null_timeteps = torch.zeros(1, device=input.device)
        output = model.post_process_net(input.permute(0,3,1,2), timesteps=null_timeteps)
        output = output.permute(0,2,3,1)

        pred_mask = torch.greater_equal(output[:, :, :, 3], 0).int()
        adjusted_img = output[:, :, :, :3]
        adjusted_img = torch.clamp(image_scaled, -1., 1.)
        adjusted_img = (adjusted_img + 1.0) / 2.0
        adjusted_img = (adjusted_img * 255).int()
        new_composite_img = adjusted_img * pred_mask.unsqueeze(3) + (1-pred_mask.unsqueeze(3)) * comp_img
        filename = img_name
        os.makedirs(save_root, exist_ok=True)
        os.makedirs(save_mask_root, exist_ok=True)
        save_path = os.path.join(save_root, filename)

        save_mask = Image.fromarray(np.array((pred_mask*255).squeeze(0).detach().cpu(), dtype=np.uint8))
        save_tuned = Image.fromarray(np.array(new_composite_img.squeeze(0).detach().cpu(), dtype=np.uint8))\
        
        img_name_1, extension = os.path.splitext(img_name)

        save_mask.save(os.path.join(save_mask_root, img_name))
        save_tuned.save(os.path.join(save_root, img_name))