import torch
# from utils.general_utils import my_knn
import torch.nn as nn
from gaussian_renderer import render
from scene import Scene, GaussianModel
import copy
import random

from torch_kdtree import build_kd_tree
import torch
from scipy.spatial import KDTree  # Reference implementation
import numpy as np
import torch.nn.functional as F
from scene.decision_mask_model import DecisionMaskModel


def prepare_merge(gaussians, merge_group_size=2, merge_num=10000, division_num=5):
    # randomly pick merge_num points, copy the gaussians, generate the new gaussians
    # and return the new gaussians

    # 1. copy the old gaussians
    # new_gaussians = copy.deepcopy(gaussians)

    # print("new_gaussians.shape\n", new_gaussians)
    # print("old_gaussians.shape\n", gaussians)

    # 2. generate random indices
    gaussians_num = gaussians.get_xyz.size(0)
    assert int(gaussians_num) > merge_num
    # random initialization
    merge_point_indices = random.sample(range(gaussians_num), merge_num)
    points_query_for_kdtree = gaussians.merged_points(merge_point_indices)

    merge_group_indices = find_merge_group(gaussians, points_query_for_kdtree, merge_group_size=merge_group_size)

    assert hasattr(gaussians, 'merge_group_indices') == False
    setattr(gaussians, "merge_group_indices", merge_group_indices)

    set_merge_weights(gaussians, merge_group_indices)
    # print("new_gaussians.shape\n", new_gaussians)
    # print("old_gaussians.shape\n", gaussians)
    return merge_point_indices


def find_merge_group(gaussian, points_query, merge_group_size=2):
    points_ref = gaussian.get_xyz
    # Create the KD-Tree on the GPU and the reference implementation
    torch_kdtree = build_kd_tree(points_ref)
    # kdtree = KDTree(points_ref.detach().cpu().numpy())

    # Search for the k nearest neighbors of each point in points_query
    k = merge_group_size
    dists, inds = torch_kdtree.query(points_query, nr_nns_searches=k)
    # dists_ref, inds_ref = kdtree.query(points_query.detach().cpu().numpy(), k=k)
    # print(f'dists: {dists}\n inds: {inds}')
    # print(f'dists_ref: {dists_ref}\n inds_ref: {inds_ref}')
    # Test for correctness
    # Note that the cupy_kdtree distances are squared
    # print(f'inds.type: {type(inds)}')
    # print(f'inds_ref.type: {type(inds_ref)}')
    # print(f'inds.shape: {(inds.shape)}')
    # print(f'inds_ref.shape: {(inds_ref.shape)}')

    # Print indices and distances for a specific query point if assertion fails
    # for i in range(len(points_query)):
    #     gpu_inds = inds[i].cpu().numpy()
    #     gpu_dists = torch.sqrt(dists[i]).detach().cpu().numpy()  # Take the square root of GPU distances
    #     cpu_inds = inds_ref[i]
    #     cpu_dists = dists_ref[i]
    #     if not np.array_equal(gpu_inds, cpu_inds):
    #         print('____________________')
    #         print(f"Query point {i} has different nearest neighbors.")
    #         print(f"GPU indices: {gpu_inds}")
    #         print(f"GPU distances: {gpu_dists}")
    #         print(f"CPU indices: {cpu_inds}")
    #         print(f"CPU distances: {cpu_dists}")
    #         print('____________________')

    # print(np.all(inds[25:30].cpu().numpy()))
    # print('********')
    # print(inds_ref[25:30])
    # assert(np.all(inds[:27].cpu().numpy() == inds_ref[:27]))
    # assert(np.allclose(torch.sqrt(dists).detach().cpu().numpy(), dists_ref, atol=1e-5))
    return inds


def set_merge_weights(gaussians, merge_group_indices):
    # init learning parameters
    merge_weight = torch.normal(mean=0.5, std=0.05, size=merge_group_indices.size(), dtype=torch.float32).to("cuda")
    merge_weight = nn.Parameter(merge_weight.requires_grad_(True))

    # decision_mask[0]: selected
    # decision_mask[1]: not selected
    # decision_mask = torch.rand((merge_group_indices.size(0), 2), dtype=torch.float32).to("cuda")
    decision_mask = DecisionMaskModel()
    # param1 = 0.2
    # param2 = 0.8
    # decision_mask[:, 0] = param1
    # decision_mask[:, 1] = param2
    # decision_mask = nn.Parameter(decision_mask.requires_grad_(True))

    # append weight to optimizer
    param_group_for_weight = {'params': [merge_weight], 'lr': 0.01,
                              "name": "merge_weight"}  # ^^^ This weight needs adjustment, maybe
    gaussians.optimizer.add_param_group(param_group_for_weight)

    param_group_for_mask = {'params': [decision_mask.parameters()], 'lr': 0.01, "name": "decision_mask"}
    # have not added the param into optimizer
    gaussians.optimizer.add_param_group(param_group_for_mask)

    assert hasattr(gaussians, 'merge_weight') == False
    setattr(gaussians, "merge_weight", merge_weight)

    assert hasattr(gaussians, 'decision_mask') == False
    setattr(gaussians, "decision_mask", decision_mask)

    # -------------print info-------------
    # print(f"total merge pairs are {len(merge_group_indices)}")
    # print(f"before merge gaussian points number is {len(gaussians.get_xyz)}")


def prepare_decision_mask_model_input(selected_groups):
    '''

    Args:
        selected_groups: gaussian groups that randomly selected

    Returns:
        model_input: processed parameters of gaussians points that will be handled by decision mask model
    '''
    print(selected_groups.size())


def calculate_new_tensor(gaussians, merge_group_size=2):
    merge_group_indices = gaussians.merge_group_indices
    assert hasattr(gaussians, 'merge_weight') == True
    assert hasattr(gaussians, 'decision_mask') == True
    # generate new gaussian pts
    assert len(gaussians.merge_weight.shape) == merge_group_size
    # decision = gumbel_softmax(gaussians.decision_mask, 0.1, True).bool()

    # decision_mask_softmax = F.softmax(gaussians.decision_mask, dim=-1)
    # decision_mask_logits = torch.log(decision_mask_softmax)
    # decision = F.gumbel_softmax(decision_mask_logits, 1, True)



    # decision_bool = decision.bool()  # 没有梯度

    decision_mask_input = prepare_decision_mask_model_input(gaussians[merge_group_indices])
    decision_bool = gaussians.decision_mask(decision_mask_input)
    # merge all points
    # decision[:, 0] = True
    # decision[:, 1] = False

    selected_idx = merge_group_indices[decision_bool[:, 0]]
    not_selected_idx = merge_group_indices[decision_bool[:, 1]]
    not_merge_mask = torch.ones(len(gaussians.get_opacity), dtype=torch.bool).to(merge_group_indices.device)
    not_merge_mask[merge_group_indices[:, 0]] = False
    not_merge_mask[merge_group_indices[:, 1]] = False
    real_pts_num = torch.sum(not_merge_mask).item()

    # selected_gaussians = gaussians.get_xyz[selected_idx]
    # not_selected_gaussians = gaussians.get_xyz[not_selected_idx]

    # print(decided_merge_indices)
    # merge_weight_broadcast = gaussians.merge_weight[:, :, None]  # merge_weight_broadcast, a, 2, 1
    # new_xyz = torch.sum(gaussians._xyz[merge_group_indices] * merge_weight_broadcast, dim=1)  # a,2,3 * a,2,1 -> a,2,3 -> a,3
    # # not equal, the above is smaller
    # # new_scaling = gaussians.scaling_inverse_activation(gaussians.get_scaling[merge_group_indices][:,0] / 0.8)
    # new_scaling = torch.sum(gaussians._scaling[merge_group_indices] * merge_weight_broadcast, dim=1) / 0.8
    # new_rotation = torch.sum(gaussians._rotation[merge_group_indices] * merge_weight_broadcast, dim=1)
    # # new_features_dc = torch.sum(gaussians._features_dc[merge_group_indices] * merge_weight_broadcast[..., None], dim=1)  # a,2,1,3 -> a,1,3
    # # new_features_rest = torch.sum(gaussians._features_rest[merge_group_indices] * merge_weight_broadcast[..., None], dim=1)  # a,2,15,3 -> a,15,3
    # new_opacity = torch.sum(gaussians._opacity[merge_group_indices] * merge_weight_broadcast, dim=1)  # a,2,1 -> a,1

    # using decision mask
    merge_weight_broadcast = gaussians.merge_weight[:, :, None]  # merge_weight_broadcast, a, 2, 1

    # TODO: mask the weights
    selected_weight = merge_weight_broadcast[decision_bool[:, 0]]
    not_selected_weight = merge_weight_broadcast[decision_bool[:, 1]]

    # handle selected gaussian groups
    selected_decision = decision[decision_bool[:, 0]]
    not_selected_decision = decision[decision_bool[:, 1]]

    # sum_of_decision = torch.sum(decision, dim=0)
    # first20_decision_mask = gaussians.decision_mask[0:20]
    # first20_decision = decision[0:20]
    # print(first20_decision_mask)
    # print(first20_decision)
    # print(sum_of_decision)
    # print(merge_weight_broadcast[:20])
    # TODO: multiply with decision mask
    new_xyz = torch.sum(gaussians._xyz[selected_idx] * selected_weight, dim=1) * (selected_decision[:, 0] * (
                1 - selected_decision[:, 1]))[:, None]  # n,3 * n -> n,3 * n,1 -> n,3
    # print(gaussians.get_xyz[selected_idx][:3])
    # print(selected_weight[:3])
    # print(new_xyz[:3])
    new_scaling = (torch.sum(gaussians._scaling[selected_idx] * selected_weight, dim=1) / 0.8) * (selected_decision[:, 0] * (1 - selected_decision[:,1]))[:, None]
    new_rotation = torch.sum(gaussians._rotation[selected_idx] * selected_weight, dim=1) * (selected_decision[:, 0] * (
                1 - selected_decision[:, 1]))[:, None]
    new_opacity = torch.sum(gaussians._opacity[selected_idx] * selected_weight, dim=1) * (selected_decision[:, 0] * (
                1 - selected_decision[:, 1]))[:, None]

    # handle not selected gaussian groups
    new_seperated_xyz = (gaussians._xyz[not_selected_idx] * not_selected_weight) * (not_selected_decision[:, 1] * (
                1 - not_selected_decision[:, 0]))[:, None, None]  # n,2,3 * n -> n,2,3 * n,1,1 -> n,2,3
    new_seperated_scaling = (gaussians._scaling[not_selected_idx] * not_selected_weight) * (not_selected_decision[:, 1] * (1 - not_selected_decision[:, 0]))[:, None, None]
    new_seperated_rotation = (gaussians._rotation[not_selected_idx] * not_selected_weight) * (not_selected_decision[:,
                                                                                              1] * (
                                                                                                          1 - not_selected_decision[
                                                                                                              :, 0]))[:,
                                                                                             None, None]
    new_seperated_opacity = (gaussians._opacity[not_selected_idx] * not_selected_weight) * (not_selected_decision[:,
                                                                                            1] * (
                                                                                                        1 - not_selected_decision[
                                                                                                            :, 0]))[:,
                                                                                           None, None]

    new_seperated_xyz = torch.cat((new_seperated_xyz[:, 0, :], new_seperated_xyz[:, 1, :]), dim=0)
    new_seperated_scaling = torch.cat((new_seperated_scaling[:, 0, :], new_seperated_scaling[:, 1, :]), dim=0)
    new_seperated_rotation = torch.cat((new_seperated_rotation[:, 0, :], new_seperated_rotation[:, 1, :]), dim=0)
    new_seperated_opacity = torch.cat((new_seperated_opacity[:, 0, :], new_seperated_opacity[:, 1, :]), dim=0)

    # concat weighted not selected points with merged selected points
    new_xyz = torch.cat((new_xyz, new_seperated_xyz), dim=0)
    new_scaling = torch.cat((new_scaling, new_seperated_scaling), dim=0)
    new_rotation = torch.cat((new_rotation, new_seperated_rotation), dim=0)
    new_opacity = torch.cat((new_opacity, new_seperated_opacity), dim=0)
    real_pts_num += len(new_xyz)
    decision_info = None

    return {"new_xyz": new_xyz,
            # "new_features_dc":new_features_dc,
            # "new_features_rest":new_features_rest,
            "new_opacity": new_opacity,
            "new_scaling": new_scaling,
            "new_rotation": new_rotation,
            "real_pts_num": real_pts_num}, decision_bool[:, 0]


def pseudo_merge_and_render(gaussians, **render_param):
    new_tensor_dict, decision_bool = calculate_new_tensor(gaussians)

    custom_data_dict = {"merge_group_indices": gaussians.merge_group_indices,
                        "new_tensor_dict": new_tensor_dict}
    # render
    render_pkg = render(render_param['viewpoint_cam'],
                        gaussians,
                        render_param['pipe'],
                        render_param['bg'],
                        custom_render=True,
                        custom_data_dict=custom_data_dict)

    return render_pkg, decision_bool


def final_merge(gaussians):
    with torch.no_grad():
        merge_group_indices = gaussians.merge_group_indices

        # do merge and obtained new points
        new_tensor_dict, decision_bool = calculate_new_tensor(gaussians)

        # delete merge_weight here to compatible with densification_postfix
        param_group_to_delete = ["merge_weight", "decision_mask"]
        for i in reversed(range(len(gaussians.optimizer.param_groups))):
            if gaussians.optimizer.param_groups[i]['name'] in param_group_to_delete:
                del gaussians.optimizer.param_groups[i]
        # for idx, param_group in enumerate(gaussians.optimizer.param_groups):
        #     # print(param_group["name"])
        #     if param_group["name"] == "merge_weight" or param_group["name"] == "decision_mask":
        #
        #         # print("Available keys in optimizer state:", gaussians.optimizer.state.keys())
        #         gaussians.optimizer.state.pop(param_group['params'][0])
        #         gaussians.optimizer.param_groups.pop(idx)
        #         # break
        assert hasattr(gaussians, 'merge_weight') == True
        assert hasattr(gaussians, 'decision_mask') == True
        del gaussians.merge_weight
        del gaussians.merge_group_indices
        del gaussians.decision_mask

        prune_filter = torch.zeros(len(gaussians.get_opacity), dtype=torch.bool).to(
            gaussians.get_opacity.device)  # shape is (n)

        # prune the merged groups of points
        prune_filter[merge_group_indices[:, 0]] = True
        prune_filter[merge_group_indices[:, 1]] = True
        gaussians.prune_points(prune_filter)

        # add resultant points
        gaussians.densification_postfix(new_xyz=new_tensor_dict["new_xyz"],
                                        # new_features_dc=new_tensor_dict["new_features_dc"],
                                        # new_features_rest=new_tensor_dict["new_features_rest"],
                                        new_opacities=new_tensor_dict["new_opacity"],
                                        new_scaling=new_tensor_dict["new_scaling"],
                                        new_rotation=new_tensor_dict["new_rotation"])

        # modify ...
        gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
        return decision_bool