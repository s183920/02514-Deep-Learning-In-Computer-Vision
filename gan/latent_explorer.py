import pickle
import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'gan/code/stylegan2-ada-pytorch-main/'))
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import PIL

# own modules
from project import run_projection
from align import align_face

class LatentExplorer:
    device = torch.device('cuda')

    def __init__(self, name) -> None:
        # with open('gan/pretrained_models/ffhq.pkl', 'rb') as f:
        with open('/zhome/c5/f/138539/02514-Deep-Learning-In-Computer-Vision/gan/pretrained_models/ffhq.pkl', 'rb') as f: # Kat bruger denne
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

        self.name = name
        self.folder = f"gan/results/{name}"
        os.makedirs(self.folder, exist_ok=True)


    def reconstruct(self, img_path, align=True):
        outdir = self.folder + "/reconstruction/" + img_path.split("/")[-1].split(".")[0]

        if align:
            img_path = align_face(img_path)

        run_projection(img_path, outdir=outdir)

    def get_latent(self, img_path, align=True, step = None):
        rep_path = self.folder + "/reconstruction/" + img_path.split("/")[-1].split(".")[0] + "/projected_w.npz"

        if os.path.exists(rep_path):
            representation =  np.load(rep_path)['w']
        else:
            self.reconstruct(img_path, align)
            representation = np.load(rep_path)['w']

        if step is None:
            return torch.from_numpy(representation).to(self.device)
        else:
            return torch.from_numpy(representation[step]).to(self.device)

    def synthesize(self, latent_rep):
        synth_image = self.G.synthesis(latent_rep.unsqueeze(0), noise_mode='const')
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        return synth_image
        # video.append_data(np.concatenate([target_uint8, synth_image], axis=1))

    def interpolate(self, img_path1, img_path2, align=True):
        z1 = self.get_latent(img_path1, step=-1, align=align)
        z2 = self.get_latent(img_path2, step=-1, align=align)
        a = 0.5
        z3 = a*z1 + (1-a)*z2
        img = self.synthesize(z3)
        return img

if __name__ == "__main__":
    le = LatentExplorer("Barbie")

    img = "gan/test_imgs/RyanGosling_Barbie.png"
    # le.reconstruct(img)


    #
    # w = le.G.mapping(le.get_latent(img)[-1], None, truncation_psi=0.5, truncation_cutoff=8)
    # img = le.synthesize(le.get_latent(img, step=-1))
    img = le.interpolate("gan/test_imgs/RyanGosling_Barbie.png", "gan/test_imgs/RyanGosling_Notebook.png")
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("gan/test.png")
