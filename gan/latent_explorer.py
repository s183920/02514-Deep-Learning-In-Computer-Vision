import pickle
import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '02514-Deep-Learning-In-Computer-Vision/gan/code/stylegan2-ada-pytorch-main/'))
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import PIL
import shutil
import imageio
from tqdm import tqdm

# own modules
from project import run_projection
from align import align_face

class LatentExplorer:
    device = torch.device('cuda')

    def __init__(self, name, align = True) -> None:
        with open('gan/pretrained_models/ffhq.pkl', 'rb') as f:
        # with open('/zhome/c5/f/138539/02514-Deep-Learning-In-Computer-Vision/gan/pretrained_models/ffhq.pkl', 'rb') as f: # Kat bruger denne
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

        self.name = name
        self.align = align
        self.folder = f"gan/results/{name}"


        # create folders
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.folder + "/original", exist_ok=True)
        os.makedirs(self.folder + "/aligned", exist_ok=True)

    def random(self, nrows = 1, ncols = 6, seed = 5):
        num_imgs = nrows * ncols

        # generate random latent vectors
        z = torch.randn(num_imgs, self.G.z_dim).cuda()

        # save images
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
        for i in range(num_imgs):
            w = self.G.mapping(z[i].unsqueeze(0), c = None, truncation_psi=0.5, truncation_cutoff=8)
            img_i = self.synthesize(w.squeeze())

            ax = axes.flatten()[i]
            ax.imshow(img_i)
            ax.axis('off')

        os.makedirs(self.folder + "/random", exist_ok=True)
        fig.tight_layout()
        plt.savefig(self.folder + "/random/random.png")

            # img_i.save(self.folder + "/original/" + str(i) + ".png")


    def split_path(self, img_path):
        img_folder, img_name = os.path.split(img_path)
        img_name, img_ext = os.path.splitext(img_name)
        return img_folder, img_name, img_ext

    def get_img_path(self, img_path):
        img_folder, img_name, img_ext = self.split_path(img_path)

        # copy image to folder
        img_path = shutil.copy(img_path, self.folder + "/original/" + img_name + img_ext)

        # align image
        if self.align:
            save_path=self.folder + "/aligned/" + img_name + img_ext
            if os.path.exists(save_path):
                return save_path
            img_path = align_face(img_path, save_path=save_path)

        return img_path

    def reconstruct(self, img_path, num_steps = 10):
        # get img names and paths
        img_folder, img_name, img_ext = self.split_path(img_path)
        img_path = self.get_img_path(img_path)

        # set output dir
        outdir = self.folder + "/reconstruction/" + img_name

        img_path = self.get_img_path(img_path)
        run_projection(img_path, outdir=outdir, num_steps=num_steps)

    def get_latent(self, img_path, step = None):
        # get img names and paths
        img_folder, img_name, img_ext = self.split_path(img_path)
        img_path = self.get_img_path(img_path)

        rep_path = self.folder + "/reconstruction/" + img_name + "/projected_w.npz"

        if os.path.exists(rep_path):
            representation =  np.load(rep_path)['w']
        else:
            self.reconstruct(img_path, num_steps=100)
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

    def interpolate(self, img_path1, img_path2, weight=0.5, plot = True, save_video = False):
        # get img names
        img_folder1, img_name1, img_ext1 = self.split_path(img_path1)
        img_folder2, img_name2, img_ext2 = self.split_path(img_path2)

        # copy to folder and align
        img_path1 = self.get_img_path(img_path1)
        img_path2 = self.get_img_path(img_path2)

        # get latent representations
        z1 = self.get_latent(img_path1, step=-1)
        z2 = self.get_latent(img_path2, step=-1)

        # interpolate
        z3 = weight*z1 + (1-weight)*z2

        # synthesize
        img = self.synthesize(z3)

        # plot
        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Interpolation with weight {weight}")
            ax[0].set_title(img_name1)
            ax[1].set_title("Interpolation")
            ax[2].set_title(img_name2)
            ax[0].imshow(PIL.Image.open(img_path1))
            ax[0].axis('off')
            ax[1].imshow(img)
            ax[1].axis('off')
            ax[2].imshow(PIL.Image.open(img_path2))
            ax[2].axis('off')

            # save plot
            os.makedirs(self.folder + "/interpolation", exist_ok=True)
            plt.savefig(self.folder + f"/interpolation/{img_name1}_to_{img_name2}.png")

        # save video
        if save_video:
            img1 = self.get_img_for_video(img_path1)
            img2 = self.get_img_for_video(img_path2)

            video = imageio.get_writer(self.folder + f'/interpolation/{img_name1}_to_{img_name2}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
            # print (f'Saving optimization progress video "{outdir}/proj.mp4"')
            for w in np.linspace(0, 1, 100):
                z3 = (1-w)*z1 + w*z2
                synth_image = self.synthesize(z3)
                video.append_data(np.concatenate([img1, synth_image, img2], axis=1))
            video.close()

        return img

    def get_img_for_video(self, img_path):
        target_pil = PIL.Image.open(img_path).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((self.G.img_resolution, self.G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        return target_uint8

    def apply_latent_dir(self, img_path, feature, magnitude = 10, feature_dir = "stylegan", save_video = False, plot = True):
        # get img names and paths
        img_folder, img_name, img_ext = self.split_path(img_path)
        img_path = self.get_img_path(img_path)

        # get feature dir
        if feature_dir == "stylegan":
            feature_dir = "gan/code/stylegan2directions/"
            feature_path = feature_dir + feature + ".npy"

            # transform feature
            w = torch.from_numpy(np.load(feature_path)).to(self.device)

        elif feature_dir == "own":
            # feature_dir = "gan/code/own_directions/"
            w = torch.from_numpy(self.load_features(feature)).to(self.device)
        else:
            raise ValueError("feature_dir must be 'stylegan' or 'own'")




        # get latent representation
        z = self.get_latent(img_path, step=-1)

         # plot
        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Latent direction: {feature}")
            ax[0].set_title("Feature reduced")
            ax[1].set_title("Original image")
            ax[2].set_title("Feature increased")
            ax[0].imshow(self.synthesize(z - magnitude*w))
            ax[0].axis('off')
            ax[1].imshow(PIL.Image.open(img_path))
            ax[1].axis('off')
            ax[2].imshow(self.synthesize(z + magnitude*w))
            ax[2].axis('off')

            # save plot
            os.makedirs(self.folder + "/latent_direction", exist_ok=True)
            plt.savefig(self.folder + f"/latent_direction/{img_name}_{feature}.{img_ext}")

        # save video
        if save_video:
            img = self.get_img_for_video(img_path)

            video = imageio.get_writer(self.folder + f'/latent_direction/{img_name}_{feature}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
            for m in np.linspace(-magnitude, magnitude, 100):
                z_trans = z + m*w
                synth_image = self.synthesize(z_trans)
                video.append_data(np.concatenate([img, synth_image], axis=1))
            video.close()

        return self.synthesize(z + magnitude*w)

    def create_features(self, feature):
        folder = f"gan/feature_dataset/{feature}"
        img_paths = folder + "/images"

        print(os.listdir(img_paths))

        for cl in os.listdir(img_paths):
            os.makedirs(folder + "/feature/" + cl, exist_ok=True)
            rec_dir = folder + "/feature/" + cl + "/latent"
            os.makedirs(rec_dir, exist_ok=True)

            print("Processing class: " + cl)
            pbar = tqdm(os.listdir(img_paths + "/" + cl))
            for img_path in pbar:
                pbar.set_description(f"Processing {img_path}")
                print(img_path)
                img_name, img_ext = img_path.split(".")
                img_path = img_paths + "/" + cl + "/" + img_path

                l_save_name = rec_dir + "/" + img_name
                if os.path.exists(l_save_name + ".npz"):
                    print(f"skipping {l_save_name} as it already exists")
                else:
                    l = run_projection(img_path, outdir=rec_dir, num_steps=100, only_latent=True)
                    np.savez(rec_dir + "/" + img_name, w = l.cpu().numpy())

    def load_features(self, feature):
        folder = f"gan/feature_dataset/{feature}"
        feature_paths = folder + "/feature"

        ws = []
        for cl in os.listdir(feature_paths):
            paths = os.listdir(feature_paths + "/" + cl + "/latent")
            f = np.zeros((len(paths), self.G.num_ws, self.G.w_dim))
            print(f.shape)
            for idx, feature_path in enumerate(paths):
                feature_path = feature_paths + "/" + cl + "/latent/" + feature_path
                # print(feature_path)

                f[idx] = np.load(feature_path)["w"]

            ws.append(f.mean(axis=0))

        return np.array(ws[0] - ws[1])



if __name__ == "__main__":
    # img paths
    img1 = "gan/test_imgs/RyanGosling_Barbie.png"
    img2 = "gan/test_imgs/RyanGosling_Notebook.png"
    # latent_dir = "gan/code/stylegan2directions/age.npy"

    # latent explorer
    le = LatentExplorer("Poster")

    # random image
    # le.random()

    # reconstruction
    # le.reconstruct(img2, num_steps=500)

    # interpolation
    # img = le.interpolate(img1, img2, save_video=True)

     # create feature
    # le.create_features("sunglasses")
    # le.load_features("sunglasses")

    # add latent direction
    # img = le.apply_latent_dir(img1, "sunglasses", feature_dir="own", save_video=True)
    # img = le.apply_latent_dir(img2, "age", save_video=True)
