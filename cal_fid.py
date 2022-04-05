# -*- coding: utf-8 -*-
# @Date    : 2/26/22
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import os
import sys
import warnings

import cma
import imageio
import numpy as np
import scipy  # noqa
import torch
import torchvision.transforms as transforms
from cleanfid.fid import compute_fid
from imageio import imsave
from PIL import Image as Image_PIL
from pytorch_pretrained_biggan import (
    convert_to_images,
)
from scipy.stats import truncnorm
from torch import nn
from tqdm import tqdm

import data_utils.utils as data_utils
import inference.utils as inference_utils
from config.config import get_cfg
from config.parser import parse_args
from data.build import build_loader

warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)
torch.manual_seed(np.random.randint(sys.maxsize))


norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_img(torch_img: torch.Tensor) -> torch.Tensor:
    """
    Take an image whose value is from -1 to 1, return a regular RGB image.
    :param torch_img: N x C x H x W.
    :return: np.ndarray (range: 0 ~ 255)
    """
    return torch_img.mul_(127.5).add_(127.5).clamp_(0.0, 255.0)


def numpy_img(torch_img: torch.Tensor) -> np.ndarray:
    return torch_img.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()


def replace_to_inplace_relu(
    model,
):  # saves memory; from https://github.com/minyoungg/pix2latent/blob/master/pix2latent/model/biggan.py
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=False))
        else:
            replace_to_inplace_relu(child)
    return


def save(out, name=None, torch_format=True):
    if torch_format:
        with torch.no_grad():
            out = out.cpu().numpy()
    img = convert_to_images(out)[0]
    if name:
        imageio.imwrite(name, np.asarray(img))
    return img


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    check_cfg(cfg)
    # default_setup(cfg, args)
    return cfg


def check_cfg(cfg):
    assert not (cfg.SOLVER.DIS_CRITIC > 1 and cfg.SOLVER.GEN_CRITIC > 1)
    assert cfg.RUN_TYPE in (
        "train",
        "finetune",
        "eval",
        "sample",
        "meta_train",
        "meta_eval",
        "meta_sample",
    )
    assert len(cfg.MODEL.LOSS.HYPER_REG.TYPE) == len(cfg.MODEL.LOSS.HYPER_REG.CONFIG)


def load_icgan(experiment_name, root_="/content"):
    root = os.path.join(root_, experiment_name)
    config = torch.load("%s/%s.pth" % (root, "state_dict_best0"))["config"]

    config["weights_root"] = root_
    config["model_backbone"] = "biggan"
    config["experiment_name"] = experiment_name
    G, config = inference_utils.load_model_inference(config)
    G.cuda()
    G.eval()
    return G


def get_output(
    noise_vector,
    input_label,
    input_features,
    stochastic_truncation,
    truncation,
    model,
    channels,
):
    if stochastic_truncation:  # https://arxiv.org/abs/1702.04782
        with torch.no_grad():
            trunc_indices = noise_vector.abs() > 2 * truncation
            size = torch.count_nonzero(trunc_indices).cpu().numpy()
            trunc = truncnorm.rvs(
                -2 * truncation, 2 * truncation, size=(1, size)
            ).astype(np.float32)
            noise_vector.data[trunc_indices] = torch.tensor(
                trunc, requires_grad=False, device="cuda"
            )
    else:
        noise_vector = noise_vector.clamp(-2 * truncation, 2 * truncation)
    if input_label is not None:
        input_label = torch.LongTensor(input_label)
    else:
        input_label = None

    out = model(
        noise_vector,
        input_label.cuda() if input_label is not None else None,
        input_features.cuda() if input_features is not None else None,
    )

    if channels == 1:
        out = out.mean(dim=1, keepdim=True)
        out = out.repeat(1, 3, 1, 1)
    return out


def load_generative_model(gen_model, last_gen_model, experiment_name, model, load_root):
    # Load generative model
    if gen_model != last_gen_model:
        model = load_icgan(experiment_name, root_=load_root)
        last_gen_model = gen_model
    return model, last_gen_model


def load_feature_extractor(gen_model, last_feature_extractor, feature_extractor):
    # Load feature extractor to obtain instance features
    feat_ext_name = "classification" if gen_model == "cc_icgan" else "selfsupervised"
    if last_feature_extractor != feat_ext_name:
        if feat_ext_name == "classification":
            feat_ext_path = ""
        else:
            # !curl - L - o / content / swav_pretrained.pth.tar - C - 'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar'
            feat_ext_path = "pretrained_models_path/swav_pretrained.pth.tar"
        last_feature_extractor = feat_ext_name
        feature_extractor = data_utils.load_pretrained_feature_extractor(
            feat_ext_path, feature_extractor=feat_ext_name
        )
        feature_extractor.eval()
    return feature_extractor, last_feature_extractor


def preprocess_input_image(input_image_path, size):
    pil_image = Image_PIL.open(input_image_path).convert("RGB")
    transform_list = transforms.Compose(
        [
            data_utils.CenterCropLongEdge(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    tensor_image = transform_list(pil_image)
    tensor_image = torch.nn.functional.interpolate(
        tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True
    )
    return tensor_image


def preprocess_generated_image(image):
    transform_list = transforms.Normalize(norm_mean, norm_std)
    image = transform_list(image * 0.5 + 0.5)
    image = torch.nn.functional.interpolate(
        image, 224, mode="bicubic", align_corners=True
    )
    return image


def main(args, save, load_root, experiment_name=None):

    cfg = setup(args)
    last_gen_model = None
    last_feature_extractor = None
    model = None
    feature_extractor = None

    size = "128"
    gen_model = "icgan"  # @param ['icgan', 'cc_icgan']
    dataset = "imagenet"  # imagenet, coco
    # if gen_model == "icgan":
    #     experiment_name = f"icgan_biggan_imagenet_res{size}"
    # else:
    #     experiment_name = f"cc_icgan_biggan_imagenet_res{size}"

    if experiment_name is None:
        if gen_model == "icgan":
            experiment_name = f"icgan_biggan_{dataset}_res{size}"
        else:
            experiment_name = f"cc_icgan_biggan_{dataset}_res{size}"

    input_image_instance = "cat.JPG"  # @param {type:"string"}
    input_feature_index = 3  # @param {type:'integer'}
    class_index = 538  # @param {type:'integer'}
    num_samples_ranked = 16  # @param {type:'integer'}
    num_samples_total = 160  # @param {type:'integer'}
    truncation = 0.7  # @param {type:'number'}
    stochastic_truncation = False  # @param {type:'boolean'}
    download_file = True  # @param {type:'boolean'}
    seed = 50  # @param {type:'number'}
    if seed == 0:
        seed = None
    noise_size = 128
    class_size = 1000
    channels = 3
    batch_size = 10
    num_imgs = 20000
    if gen_model == "icgan":
        class_index = None

    assert num_samples_ranked <= num_samples_total
    import numpy as np

    state = None if not seed else np.random.RandomState(seed)
    np.random.seed(seed)

    feature_extractor_name = (
        "classification" if gen_model == "cc_icgan" else "selfsupervised"
    )

    # Load feature extractor (outlier filtering and optionally input image feature extraction)
    feature_extractor, last_feature_extractor = load_feature_extractor(
        gen_model, last_feature_extractor, feature_extractor
    )

    # Load generative model
    model, last_gen_model = load_generative_model(
        gen_model, last_gen_model, experiment_name, model, load_root
    )
    # Prepare other variables
    name_file = "%s_class_index%s_instance_index%s" % (
        gen_model,
        str(class_index) if class_index is not None else "None",
        str(input_feature_index) if input_feature_index is not None else "None",
    )

    replace_to_inplace_relu(model)

    # Load features
    print("Obtaining instance features from input image!")
    input_feature_index = None
    # input_image_tensor = preprocess_input_image(input_image_instance, int(size))
    # print("Displaying instance conditioning:")
    # display(
    #     convert_to_images(
    #         ((input_image_tensor * norm_std + norm_mean) - 0.5) / 0.5
    #     )[0]
    # )
    # data loader
    dataset_name = cfg.DATASETS.TEST[0]
    print(f"dataset_name: {dataset_name}")
    support_loader_iter = iter(
        build_loader(
            cfg,
            dataset_name,
            batch_size,
            cfg.DATALOADER.NUM_WORKERS,
            is_train=False,
            continuous_class_id=None,
        )
    )

    num_iter = max(num_imgs // (batch_size * cfg.META_LEARN.NUM_SUPPORT_SHOT), 1)
    all_outs, all_dists = [], []
    for _ in tqdm(range(num_iter), desc=dataset_name):
        data_dict = next(support_loader_iter)
        num_task, num_support_shot, c, h, w = data_dict["support_set"].shape
        assert num_task == batch_size
        support_imgs = (
            data_dict["support_set"]
            .reshape(num_task * num_support_shot, c, h, w)
            .type(torch.cuda.FloatTensor)
        )
        # Create noise, instance and class vector
        noise_vector = truncnorm.rvs(
            -2 * truncation,
            2 * truncation,
            size=(num_samples_total, noise_size),
            random_state=state,
        ).astype(
            np.float32
        )  # see https://github.com/tensorflow/hub/issues/214
        noise_vector = torch.tensor(noise_vector, requires_grad=False, device="cuda")

        with torch.no_grad():
            input_features, _ = feature_extractor(support_imgs.cuda())
        # input_features /= torch.linalg.norm(input_features, dim=-1, keepdims=True)
        # instance_vector = torch.tensor(
        #     input_features, requires_grad=False, device="cuda"
        # ).repeat(num_samples_total, 1)

        input_label = None
        if input_feature_index is not None:
            print("Conditioning on instance with index: ", input_feature_index)

        size = int(size)

        for task_idx in range(num_task):
            start = task_idx * num_support_shot
            end = (task_idx + 1) * num_support_shot
            cur_instance_vector = input_features[start:end] / torch.linalg.norm(
                input_features[start:end], dim=-1, keepdims=True
            )
            out = get_output(
                noise_vector[start:end],
                input_label[start:end] if input_label is not None else None,
                cur_instance_vector.cuda() if cur_instance_vector is not None else None,
                stochastic_truncation,
                truncation,
                model,
                channels,
            )

            all_outs.extend(list(numpy_img(denormalize_img(out))))
            # all_outs.append(out.detach().cpu())
            del out
    fid_score = compute_fid(
        dataset_name=dataset_name.replace("meta_", ""),
        dataset_res=cfg.DATASETS.RESOLUTION,
        np_images=all_outs,
        dataset_split="custom" if "cifar100" in dataset_name else "xinyu",
    )
    print(f"{dataset_name}'s fid score: {fid_score}")

    if save:
        output_dir = f"out_{experiment_name}_{dataset_name}"
        os.makedirs(output_dir)
        print(f"saving to {output_dir}")
        for im_idx, im in enumerate(tqdm(all_outs)):
            file_name = os.path.join(output_dir, f"sample_{im_idx}.png")
            imsave(file_name, im)


if __name__ == "__main__":
    save = False
    args = parse_args()
    print("Command Line Args:", args)
    load_root = "pretrained_models_path"
    experiment_name = None
    main(args, save, load_root=load_root, experiment_name=experiment_name)
