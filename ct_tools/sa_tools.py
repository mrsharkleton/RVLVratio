
import json
from enum import IntEnum
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from scipy.spatial import distance
from scipy.ndimage import center_of_mass, binary_dilation, distance_transform_edt
import cv2
from scipy.spatial import cKDTree
# set a non-interactive backend for matplotlib
plt.switch_backend('Agg')

class CardiacLabel(IntEnum):
    LV_BP  = 1
    LV_MYO = 2
    RV_BP  = 3
    RV_MYO = 4
    LA     = 5
    RA     = 6
    AAo    = 7
    PA     = 8
    DAo    = 9

class SADiameterTool():

    def __init__(self, image_path, label_path, output_path=None ):
        self.image_path = image_path
        self.label_path = label_path
        self.output_path = output_path
        self.desired_labels = [3,4,5,6,7,8,9,10,11]
        self.fig, self.ax = plt.subplots(2, 4, figsize=(12, 6), dpi=300, constrained_layout=True)
        self.aha_fig = plt.figure(figsize=(9, 6), layout="constrained", dpi=300) # have to do it differently for polar plots
        self.aha_fig.get_layout_engine().set(wspace=0.1, w_pad=0.2)
        self.aha_ax_gs = self.aha_fig.add_gridspec(2, 3)

        base = plt.get_cmap("tab10")
        colours = np.vstack([
            [0,0,0,0],
            base(np.linspace(0, 1, len(CardiacLabel)))
        ])
        self.cmap_10segs = ListedColormap(colours)
        base = plt.get_cmap("tab20")
        colours = np.vstack([
            [0,0,0,0],
            base(np.linspace(0, 1, 17))
        ])
        self.cmap_17segs = ListedColormap(colours)
        self.cm_ll = -150
        self.cm_ul = 250
        self.fontsize = 8
        self.process()


    def process(self):

        self.read_image_and_label()
        self.get_np_cardiac_masks()
        self.calculate_lv_mid_point()
        self.calculate_pca_major_axis()
        self.crop_to_cardiac_sa()
        self.select_mid_lv_slices()
        self.calculate_septal_angle_and_diameters()
        self.plot_septal_angle_and_diameters()

        if self.output_path:
            self.export_figure()
        plt.close(self.fig)

        self.thickness_analysis()

    def thickness_analysis(self):

        self.select_aha_slices()
        self.calculate_aha_slices()
        thickness_results = self.structure_thickness_analysis(endocardial_mask=self.lv_endo_mask_sa, epicardial_mask=self.lv_epi_mask_sa, image=self.img_sa)

        mean_thickness = []
        p10_thickness = []
        annotations_mean = {}
        annotations_p10 = {}
        for i in range(1, 18):
            seg_thickness = thickness_results[i]["mean_thickness_mm"]
            p10 = thickness_results[i]["p10_thickness_mm"]
            std = thickness_results[i]["stddev_thickness_mm"]

            mean_thickness.append(seg_thickness)
            p10_thickness.append(p10)
            annotations_mean[i] = f"{seg_thickness:.1f}\n±{std:.1f}"
            annotations_p10[i] = f"{p10:.1f}"
            

        ax_bull_mean = self.aha_fig.add_subplot(self.aha_ax_gs[0,2], projection='polar')
        ax_bull_mean.set_title("Mean LV Wall Thickness (mm)", fontsize=self.fontsize+2)
        ax_bull_mean_norm = mpl.colors.Normalize(
                vmin=6.0,   # thin but plausible
                vmax=14.0   # hypertrophied
            )
        self.bullseye_plot(ax=ax_bull_mean, data=mean_thickness, segment_stats=annotations_mean, cmap=mpl.cm.viridis, norm=ax_bull_mean_norm)

        ax_bull_p10 = self.aha_fig.add_subplot(self.aha_ax_gs[1,2], projection='polar')
        ax_bull_p10.set_title("10th Percentile LV Wall Thickness (mm)", fontsize=self.fontsize+2)
        ax_bull_p10_norm = mpl.colors.TwoSlopeNorm(
            vmin=3.0,   # severe thinning
            vcenter=6.5,  # lower limit of normal
            vmax=10.0   # safely normal
        )
        self.bullseye_plot(ax=ax_bull_p10, data=p10_thickness, segment_stats=annotations_p10, cmap=mpl.cm.RdBu_r, norm=ax_bull_p10_norm)
        # self.bullseye_plot_orig(ax=ax_bull, data=mean_thickness)#, segment_stats=annotations)

        # save SA image and label
        if self.output_path:
            sitk.WriteImage(self.img_sa, self.output_path.replace('.png', '_sa_image.nii.gz'))
            sitk.WriteImage(self.lbl_sa, self.output_path.replace('.png', '_sa_label.nii.gz'))
            sitk.WriteImage(self.aha_lv_myo_img, self.output_path.replace('.png', '_sa_aha_lv_myo.nii.gz'))
            json.dump(thickness_results, open(self.output_path.replace('.png', '_thickness.json'), 'w'), indent=4)
            aha_output_path = self.output_path.replace('.png', '_aha.png')
            self.aha_fig.savefig(aha_output_path, dpi=300)
            plt.close(self.aha_fig)



    def export_figure(self):
        self.fig.savefig(self.output_path, dpi=300)
        json_path = self.output_path.replace('.png', '.json')
        json_dict = {
            "mean_septal_angle_degrees": self.mean_sep_angle,
            "mean_lv_diameter_mm": self.mean_lv_diam,
            "mean_rv_diameter_mm": self.mean_rv_diam,
            "mean_rv_lv_ratio": self.mean_rv_lv_ratio,
            "septal_angles_degrees": self.sep_angles,
            "lv_diameters_mm": self.lv_diams,
            "rv_diameters_mm": self.rv_diams,
            "rv_lv_ratios": self.rv_lv_ratios,
            "A_world_to_cardiac": self.A_world_to_cardiac.tolist(),
            "A_cardiac_to_world": self.A_cardiac_to_world.tolist(),
        }
        json.dump(json_dict, open(json_path, 'w'), indent=4)

    @staticmethod
    def bullseye_plot_orig(ax, data, seg_bold=None, cmap="cividis", norm=None):
        """
        Bullseye representation for the left ventricle.

        Parameters
        ----------
        ax : Axes
        data : list[float]
            The intensity values for each of the 17 segments.
        seg_bold : list[int], optional
            A list with the segments to highlight.
        cmap : colormap, default: "cividis"
            Colormap for the data.
        norm : Normalize or None, optional
            Normalizer for the data.

        Notes
        -----
        This function creates the 17 segment model for the left ventricle according
        to the American Heart Association (AHA) [1]_

        References
        ----------
        .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
            S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
            and M. S. Verani, "Standardized myocardial segmentation and
            nomenclature for tomographic imaging of the heart",
            Circulation, vol. 105, no. 4, pp. 539-542, 2002.
        """
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        data = np.ravel(data)
        if seg_bold is None:
            seg_bold = []
        if norm is None:
            norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

        r = np.linspace(0.2, 1, 4)

        ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
        ax.grid(False)  # Remove grid

        # Fill segments 1-6, 7-12, 13-16.
        for start, stop, r_in, r_out in [
                (0, 6, r[2], r[3]),
                (6, 12, r[1], r[2]),
                (12, 16, r[0], r[1]),
                (16, 17, 0, r[0]),
        ]:
            n = stop - start
            dtheta = 2*np.pi / n
            ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
                color=cmap(norm(data[start:stop])))

        # Now, draw the segment borders.  In order for the outer bold borders not
        # to be covered by inner segments, the borders are all drawn separately
        # after the segments have all been filled.  We also disable clipping, which
        # would otherwise affect the outermost segment edges.
        # Draw edges of segments 1-6, 7-12, 13-16.
        for start, stop, r_in, r_out in [
                (0, 6, r[2], r[3]),
                (6, 12, r[1], r[2]),
                (12, 16, r[0], r[1]),
        ]:
            n = stop - start
            dtheta = 2*np.pi / n
            ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
                clip_on=False, color="none", edgecolor="k", linewidth=[
                    4 if i + 1 in seg_bold else 2 for i in range(start, stop)])
        # Draw edge of segment 17 -- here; the edge needs to be drawn differently,
        # using plot().
        ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
                linewidth=(4 if 17 in seg_bold else 2))

        cbar = ax.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='vertical', fraction=0.02, pad=0.04, location='right')

    def read_image_and_label(self):
        img = sitk.ReadImage(self.image_path, sitk.sitkFloat32)
        lbl = sitk.ReadImage(self.label_path, sitk.sitkUInt64)
        lbl_np = sitk.GetArrayFromImage(lbl)
        # Remap our bitmask labels to a compact range starting from 1
        remapped_lbl = np.zeros_like(lbl_np, dtype=np.uint8)
        for new_label, old_label in enumerate(self.desired_labels, start=1):
            # remapped_lbl[(lbl_np >> old_label) & 1] = new_label
            remapped_lbl |= (((lbl_np >> old_label) & 1) * new_label).astype(np.uint8)
        remapped_lbl_img = sitk.GetImageFromArray(remapped_lbl)
        remapped_lbl_img.CopyInformation(lbl)

        self.img = img
        self.lbl = remapped_lbl_img


    def get_np_cardiac_masks(self):
        self.lbl_np = sitk.GetArrayFromImage(self.lbl)  # Z,Y,X

        self.lv_mask_oo = (self.lbl_np == CardiacLabel.LV_MYO) | (self.lbl_np == CardiacLabel.LV_BP)
        self.rv_mask_oo = (self.lbl_np == CardiacLabel.RV_MYO) | (self.lbl_np == CardiacLabel.RV_BP)
        self.la_mask_oo = (self.lbl_np == CardiacLabel.LA)
        self.cardiac_mask_oo = self.lv_mask_oo | self.rv_mask_oo | self.la_mask_oo | (self.lbl_np == CardiacLabel.RA)


    def calculate_lv_mid_point(self):
        lv_cog_vox = np.mean(np.column_stack(np.where(self.lv_mask_oo)),axis=0)  # Z,Y,X
        self.lv_cog_mm_world = self.img.TransformIndexToPhysicalPoint((int(lv_cog_vox[2]), int(lv_cog_vox[1]), int(lv_cog_vox[0])))

        # let's display the LV COG in the current image space
        img_np = sitk.GetArrayFromImage(self.img)
        self.ax[0,0].imshow(img_np[ int(lv_cog_vox[0]) ], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
        self.ax[0,0].scatter([lv_cog_vox[2]], [lv_cog_vox[1]], c="b", s=1)
        self.ax[0,0].set_title("LV COG\n (original image space)", fontsize=self.fontsize+1)
        self.ax[0,0].axis("off")
    

    def calculate_pca_major_axis(self):
        # get our mask coordinates in physical space
        lv_pts = self.mask_to_physical_points(self.lv_mask_oo, self.img)
        rv_pts = self.mask_to_physical_points(self.rv_mask_oo, self.img)
        la_pts = self.mask_to_physical_points(self.la_mask_oo, self.img)

        # PCA on LV points to get long axis direction
        pca = PCA(n_components=3)
        pca.fit(lv_pts)
        z_hat = pca.components_[0]
        z_hat /= np.linalg.norm(z_hat)

        # centres of gravity in physical space
        cog_lv = lv_pts.mean(axis=0)
        cog_la = la_pts.mean(axis=0)
        cog_rv = rv_pts.mean(axis=0)

        # ensure z_hat points from LV to LA
        if np.dot(z_hat, cog_la - cog_lv) < 0:
            z_hat = -z_hat

        # define x_hat as the projection of LV->RV vector onto plane normal to z_hat
        lv_to_rv = cog_rv - cog_lv
        x_hat = lv_to_rv - np.dot(lv_to_rv, z_hat) * z_hat
        x_hat /= np.linalg.norm(x_hat)

        # ensure x_hat points roughly in the positive X direction
        if np.dot(x_hat, np.array([1,0,0])) < 0:
            x_hat = -x_hat
        
        # define y_hat to complete right-handed system
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= np.linalg.norm(y_hat)

        # re-orthogonalise
        x_hat = np.cross(y_hat, z_hat)
        R = np.vstack([x_hat, y_hat, z_hat])

        # transformation matrices
        self.A_world_to_cardiac = np.eye(4)
        self.A_world_to_cardiac[:3,:3] = R
        self.A_world_to_cardiac[:3, 3] = -R @ cog_lv

        # the actual one we need is the inverse
        self.A_cardiac_to_world = np.linalg.inv(self.A_world_to_cardiac)


    def crop_to_cardiac_sa(self):

        # let's limit it to the cardiac region
        cariac_indices = np.column_stack(np.where(self.cardiac_mask_oo))  # Z,Y,X
        max_cardiac_corners = cariac_indices.max(axis=0)
        min_cardiac_corners = cariac_indices.min(axis=0)

        cardiac_corners = np.array([
            [min_cardiac_corners[2], min_cardiac_corners[1], min_cardiac_corners[0]],
            [max_cardiac_corners[2], min_cardiac_corners[1], min_cardiac_corners[0]],
            [min_cardiac_corners[2], max_cardiac_corners[1], min_cardiac_corners[0]],
            [min_cardiac_corners[2], min_cardiac_corners[1], max_cardiac_corners[0]],
            [max_cardiac_corners[2], max_cardiac_corners[1], min_cardiac_corners[0]],
            [max_cardiac_corners[2], min_cardiac_corners[1], max_cardiac_corners[0]],
            [min_cardiac_corners[2], max_cardiac_corners[1], max_cardiac_corners[0]],
            [max_cardiac_corners[2], max_cardiac_corners[1], max_cardiac_corners[0]],
        ])

        corners_phys = np.array([
            self.img.TransformIndexToPhysicalPoint((int(c[0]), int(c[1]), int(c[2])))
            for c in cardiac_corners
        ])

        corners_cardiac = (self.A_world_to_cardiac[:3,:3] @ corners_phys.T).T + self.A_world_to_cardiac[:3,3]
        min_c = corners_cardiac.min(axis=0)
        max_c = corners_cardiac.max(axis=0)
        out_spacing = self.img.GetSpacing()
        extent = max_c - min_c
        out_size = np.ceil(extent / out_spacing).astype(int)

        out_origin = min_c
        out_direction = np.eye(3).ravel()

        # enforce square in-plane spacing explicitly
        out_spacing = list(out_spacing)
        out_spacing[1] = out_spacing[0]   # Y = X Exception has occurred: TypeError 'tuple' object does not support item assignment

        tx = sitk.AffineTransform(3)
        tx.SetMatrix(self.A_cardiac_to_world[:3,:3].ravel())
        tx.SetTranslation(self.A_cardiac_to_world[:3,3])

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(out_size.tolist())
        resampler.SetOutputSpacing(out_spacing)
        resampler.SetOutputOrigin(out_origin.tolist())
        resampler.SetOutputDirection(out_direction)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(tx)
        resampler.SetDefaultPixelValue(-1024)

        self.img_sa = resampler.Execute(self.img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        self.lbl_sa = resampler.Execute(self.lbl)


    def select_aha_slices(self):
        """ Selects the short-axis slices in which the LV myocardium formes a closed circumferential ring, excluding basal valve-plane slices """

        # 1 start with the valid_slices_closed_cavity from select_mid_lv_slices
        # self.select_mid_lv_slices()

        # 2. fill in any gaps in the valid slices and add in the apical cap slices
        all_slices = list(set(range(self.valid_slices_closed_cavity[0], self.valid_slices_closed_cavity[-1] +1)))

        z_slices_with_lv = np.where(self.lv_epi_mask_sa.sum(axis=(1,2)) > 0)[0].tolist()

        last_slice = all_slices[0]
        max_slice = z_slices_with_lv[0]
        apical_cap_slices = []

        for z in range(last_slice-1, max_slice-1, -1):
            myo = self.lv_epi_mask_sa[z]
            bp = self.lv_endo_mask_sa[z]

            if myo.sum() > 0 and bp.sum() > 0:
                # still have LV cavity
                all_slices.append(z)
            elif myo.sum() > 0 and bp.sum() == 0:
                # no more LV cavity
                apical_cap_slices.append(z)
    
        self.apical_cap_slices = apical_cap_slices

        # the other slices get split into thirds, basal, mid and apical

        third = len(all_slices) // 3
        self.basal_slices = all_slices[2*third:]
        self.mid_slices = all_slices[third:2*third]
        self.apical_slices = all_slices[:third]

    
    def calculate_aha_slices(self):
        """ Plot the selected AHA slices """

        # I think we possibly need to do our algebra again in the cropped SA image space

        mask_as_points = self.mask_to_physical_points(self.lv_endo_mask_sa, self.img_sa)
        centre_point_mm = np.mean(mask_as_points, axis=0)
        (Z, Y, X) = self.img_sa.TransformPhysicalPointToIndex(centre_point_mm)  # in SA image space, Z,Y,X



        aha_mask = np.zeros_like(self.lbl_sa_np, dtype=np.uint8)
        for z in self.basal_slices:
            lv_bp_mask = self.lv_endo_mask_sa[z]
            cog = np.mean(np.argwhere(lv_bp_mask), axis=0)  # Y,X
            aha_mask[z] = self.create_aha_masks(aha_mask[z].shape, centre_point=(cog[1], cog[0]), ref_vector=(1,0), location="basal") # ref vector is pointing to the left
        for z in self.mid_slices:
            lv_bp_mask = self.lv_endo_mask_sa[z]
            cog = np.mean(np.argwhere(lv_bp_mask), axis=0)  # Y,X
            aha_mask[z] = self.create_aha_masks(aha_mask[z].shape, centre_point=(cog[1], cog[0]), ref_vector=(1,0), location="mid") # ref vector is pointing to the left
        for z in self.apical_slices:
            lv_bp_mask = self.lv_endo_mask_sa[z]
            cog = np.mean(np.argwhere(lv_bp_mask), axis=0)  # Y,X
            aha_mask[z] = self.create_aha_masks(aha_mask[z].shape, centre_point=(cog[1], cog[0]), ref_vector=(1,0), location="apical") # ref vector is pointing to the left
        for z in self.apical_cap_slices:
            lv_endo_mask = self.lv_endo_mask_sa[z]
            cog = np.mean(np.argwhere(lv_endo_mask), axis=0)  # Y,X
            aha_mask[z] = self.create_aha_masks(aha_mask[z].shape, centre_point=(cog[1], cog[0]), ref_vector=(1,0), location="apex_cap") # ref vector is pointing to the left
        
        self.aha_mask = aha_mask
        # let's transform our LV myocardium label to AHA segments
        self.aha_lv_myo = self.aha_mask * (self.lbl_sa_np == CardiacLabel.LV_MYO)
        # next convert to a sitk image/label with correct info
        self.aha_lv_myo_img = sitk.GetImageFromArray(self.aha_lv_myo.astype(np.uint8))
        self.aha_lv_myo_img.CopyInformation(self.lbl_sa)
        # basal slice
        if len(self.basal_slices) > 0:
            basal_slice = self.basal_slices[len(self.basal_slices)//2]
            basal_ax = self.aha_fig.add_subplot(self.aha_ax_gs[0,0])
            basal_ax.imshow(self.img_sa_np[basal_slice], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            basal_ax.imshow(self.aha_lv_myo[basal_slice], cmap=self.cmap_17segs, vmin=0, vmax=len(self.cmap_17segs.colors), alpha=0.7)
            basal_ax.set_title("LV SA Basal LV", fontsize=self.fontsize+1)
            basal_ax.axis("off")
        # mid slice
        if len(self.mid_slices) > 0:
            mid_slice = self.mid_slices[len(self.mid_slices)//2]
            ax_mid = self.aha_fig.add_subplot(self.aha_ax_gs[0,1])
            ax_mid.imshow(self.img_sa_np[mid_slice], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            ax_mid.imshow(self.aha_lv_myo[mid_slice], cmap=self.cmap_17segs, vmin=0, vmax=len(self.cmap_17segs.colors), alpha=0.7)
            ax_mid.set_title("LV SA Mid LV", fontsize=self.fontsize+1)
            ax_mid.axis("off")
        # apical slices
        if len(self.apical_slices) > 0:
            apical_slice = self.apical_slices[len(self.apical_slices)//2]
            ax_apical = self.aha_fig.add_subplot(self.aha_ax_gs[1,0])
            ax_apical.imshow(self.img_sa_np[apical_slice], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            ax_apical.imshow(self.aha_lv_myo[apical_slice], cmap=self.cmap_17segs, vmin=0, vmax=len(self.cmap_17segs.colors), alpha=0.7)
            ax_apical.set_title("LV SA Apical LV", fontsize=self.fontsize+1)
            ax_apical.axis("off")
        
        # apical cap slices
        if len(self.apical_cap_slices) > 0:
            apical_cap_slice = self.apical_cap_slices[len(self.apical_cap_slices)//2]
            ax_apical_cap = self.aha_fig.add_subplot(self.aha_ax_gs[1,1])
            ax_apical_cap.imshow(self.img_sa_np[apical_cap_slice], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            ax_apical_cap.imshow(self.aha_lv_myo[apical_cap_slice], cmap=self.cmap_17segs, vmin=0, vmax=len(self.cmap_17segs.colors), alpha=0.7)
            ax_apical_cap.set_title("LV SA Apical Cap", fontsize=self.fontsize+1)
            ax_apical_cap.axis("off")

        # uncomment to add colorbar for AHA segments
        # cbar = self.aha_fig.colorbar(plt.cm.ScalarMappable(cmap=self.cmap_17segs, norm=plt.Normalize(vmin=0, vmax=len(self.cmap_17segs.colors))), ax=[basal_ax, ax_mid, ax_apical, ax_apical_cap], orientation='vertical', fraction=0.02, pad=0.04, location='left')


    def select_mid_lv_slices(self):
        """Calculate short-axis slices through the LV. We want to avoid too apical or basal slices."""
        self.img_sa_np = sitk.GetArrayFromImage(self.img_sa)
        self.lbl_sa_np = sitk.GetArrayFromImage(self.lbl_sa)
        # smaller cropped masks:
        self.lv_endo_mask_sa = (self.lbl_sa_np == CardiacLabel.LV_BP)
        self.lv_epi_mask_sa = (self.lbl_sa_np == CardiacLabel.LV_MYO) | (self.lbl_sa_np == CardiacLabel.LV_BP)
        self.rv_endo_mask_sa = (self.lbl_sa_np == CardiacLabel.RV_BP)

        all_lv_indices = np.column_stack(np.where(self.lv_endo_mask_sa))  # Z,Y,X
        z_min = all_lv_indices[:,0].min()
        z_max = all_lv_indices[:,0].max()

        # slices to consider around mid-LV
        self.valid_slices_closed_cavity = []
        for z in range(z_min, z_max+1):
            bp = self.lv_endo_mask_sa[z]
            myo = self.lv_epi_mask_sa[z]

            if bp.sum() == 0:
                continue

            # dilate BP and see if it escapes myocardium
            bp_dil = binary_dilation(bp, iterations=2)
            if np.all(bp_dil <= myo):
                self.valid_slices_closed_cavity.append(z)

        # next to avoid valves and apical tapering, we pick the slice with the largest LV cavity area
        # lv_areas = lv_endo_mask.sum(axis=(1,2))

        num_valid_slices = len(self.valid_slices_closed_cavity)
        if num_valid_slices == 0:
            raise ValueError("No valid LV slices with closed cavity found.")
        
        middle_two_thirds = int(0.66 * num_valid_slices)
        lower_bound = int((num_valid_slices - middle_two_thirds) / 2)
        upper_bound = lower_bound + middle_two_thirds
        valid_slices = self.valid_slices_closed_cavity[lower_bound:upper_bound] # these are going from apical to basal

        # “Measurements were performed on mid-ventricular short-axis slices selected based on LV cavity area and enclosure by myocardium to avoid basal valve-plane and apical tapering effects.”

        # let's show the lower, upper and mid slices
        if len(valid_slices) == 0:
            raise ValueError("No valid mid-LV slices found.")
        elif len(valid_slices) == 1:
            self.ax[0,1].imshow(self.img_sa_np[valid_slices[0]], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            self.ax[0,1].imshow(self.lbl_sa_np[valid_slices[0]], cmap=self.cmap_10segs, vmin=0, vmax=len(self.cmap_10segs.colors), alpha=0.3)
            self.ax[0,1].set_title("LV SA Mid LV", fontsize=self.fontsize+1)
            self.ax[0,1].axis("off")
        elif len(valid_slices) == 2:
            self.ax[0,1].imshow(self.img_sa_np[valid_slices[0]], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            self.ax[0,1].imshow(self.lbl_sa_np[valid_slices[0]], cmap=self.cmap_10segs, vmin=0, vmax=len(self.cmap_10segs.colors), alpha=0.3)
            self.ax[0,1].set_title("LV SA Apical LV", fontsize=self.fontsize+1)
            self.ax[0,1].axis("off")
            self.ax[0,2].imshow(self.img_sa_np[valid_slices[1]], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            self.ax[0,2].imshow(self.lbl_sa_np[valid_slices[1]], cmap=self.cmap_10segs, vmin=0, vmax=len(self.cmap_10segs.colors), alpha=0.3)
            self.ax[0,2].set_title("LV SA Basal LV", fontsize=self.fontsize+1)
            self.ax[0,2].axis("off")
        else:
            self.ax[0,1].imshow(self.img_sa_np[valid_slices[0]], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            self.ax[0,1].imshow(self.lbl_sa_np[valid_slices[0]], cmap=self.cmap_10segs, vmin=0, vmax=len(self.cmap_10segs.colors), alpha=0.3)
            self.ax[0,1].set_title("LV SA Apical LV", fontsize=self.fontsize+1)
            self.ax[0,1].axis("off")
            self.ax[0,2].imshow(self.img_sa_np[valid_slices[len(valid_slices)//2]], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            self.ax[0,2].imshow(self.lbl_sa_np[valid_slices[len(valid_slices)//2]], cmap=self.cmap_10segs, vmin=0, vmax=len(self.cmap_10segs.colors), alpha=0.3)
            self.ax[0,2].set_title("LV SA Mid LV", fontsize=self.fontsize+1)
            self.ax[0,2].axis("off")
            self.ax[0,3].imshow(self.img_sa_np[valid_slices[-1]], cmap="gray", vmin=self.cm_ll, vmax=self.cm_ul)
            self.ax[0,3].imshow(self.lbl_sa_np[valid_slices[-1]], cmap=self.cmap_10segs, vmin=0, vmax=len(self.cmap_10segs.colors), alpha=0.3)
            self.ax[0,3].set_title("LV SA Basal LV", fontsize=self.fontsize+1)
            self.ax[0,3].axis("off")
        
        self.lv_slices = valid_slices


    def calculate_septal_angle_and_diameters(self):
        """Calculate septal angle and RV/LV diameters for selected LV slices."""
        results = [] #apical to basal
        for sli_z in self.lv_slices:
            sep_data = self.extract_septum(self.lv_endo_mask_sa[sli_z].astype(np.uint8), self.lv_epi_mask_sa[sli_z].astype(np.uint8), self.rv_endo_mask_sa[sli_z].astype(np.uint8))
            # diam_data = self.calculate_rvlv_diameter_horizontal(self.lv_endo_mask[sli_z], self.rv_endo_mask[sli_z], sli_z)
            diam_data = self.calculate_rvlv_diameter_perpendicular_to_septum(self.lv_endo_mask_sa[sli_z], self.rv_endo_mask_sa[sli_z], sep_data, sli_z)
            result = {**sep_data, **diam_data}
            results.append(result)
        
        self.septal_diameter_results = results


    def plot_septal_angle_and_diameters(self):
        """Here we calculate some mean values and plot the apical, mid and basal slices with septal angle annotation.
           We also generate graphs of the LV angle and RV/LV diameters vs slice number."""

        lv_slices = self.lv_slices
        self.sep_angles = [res["septum_angle"] for res in self.septal_diameter_results]
        self.lv_diams = [res["lv_diameter_mm"] for res in self.septal_diameter_results]
        self.rv_diams = [res["rv_diameter_mm"] for res in self.septal_diameter_results]
        self.rv_lv_ratios = [res["rv_lv_ratio"] for res in self.septal_diameter_results]
        self.mean_sep_angle = np.mean(self.sep_angles)
        self.mean_lv_diam = np.mean(self.lv_diams)
        self.mean_rv_diam = np.mean(self.rv_diams)
        self.mean_rv_lv_ratio = np.mean(self.rv_lv_ratios)

        # we want a matchin array centred on zero showing the distance along the LV long axis
        slice_thickness = self.img_sa.GetSpacing()[2]
        self.slice_positions = [s * slice_thickness for s in range(len(lv_slices))]
        self.slice_positions = [pos - np.mean(self.slice_positions) for pos in self.slice_positions]

        if len(lv_slices) == 0:
            return
        if len(lv_slices) == 1:
            # only one slice
            ax_single = self.ax[0,1]
            self.plot_slice_with_septal_angle(
                ax_single,
                self.septal_diameter_results[0],
                fontsize=self.fontsize
            )

        elif len(lv_slices) == 2:
            # apical slice
            ax_apical = self.ax[0,1]
            self.plot_slice_with_septal_angle(
                ax_apical,
                self.septal_diameter_results[0],
                fontsize=self.fontsize
            )

            # basal slice
            ax_basal = self.ax[0,2]
            self.plot_slice_with_septal_angle(
                ax_basal,
                self.septal_diameter_results[1],
                fontsize=self.fontsize
            )

        elif len(lv_slices) >= 3:
            # apical slice
            ax_apical = self.ax[0,1]
            self.plot_slice_with_septal_angle(
                ax_apical,
                self.septal_diameter_results[0],
                fontsize=self.fontsize
            )

            # mid slice
            ax_mid = self.ax[0,2]
            mid_idx = len(lv_slices) // 2
            self.plot_slice_with_septal_angle(
                ax_mid,
                self.septal_diameter_results[mid_idx],
                fontsize=self.fontsize
            )

            # basal slice
            ax_basal = self.ax[0,3]
            self.plot_slice_with_septal_angle(
                ax_basal,
                self.septal_diameter_results[-1],
                fontsize=self.fontsize
            )
        # plots of angle and diameters
        self.plot_line_graph(
            self.ax[1,0],
            self.slice_positions,
            self.sep_angles,
            xlabel="Distance along LV long axis (mm)",
            ylabel="Septal Angle (degrees)",
            title="Septal Angle vs Distance",
            colour="aqua",
            text=f'Mean Septal Angle: {self.mean_sep_angle:.2f}°',
            fontsize=self.fontsize
        )
        self.plot_line_graph(
            self.ax[1,1],
            self.slice_positions,
            self.lv_diams,
            xlabel="Distance along LV long axis (mm)",
            ylabel="LV Diameter (mm)",
            title="LV Diameter vs Distance",
            colour="blue",
            text=f'Mean LV Diameter: {self.mean_lv_diam:.2f} mm',
            fontsize=self.fontsize
        )
        self.plot_line_graph(
            self.ax[1,2],
            self.slice_positions,
            self.rv_diams,
            xlabel="Distance along LV long axis (mm)",
            ylabel="RV Diameter (mm)",
            title="RV Diameter vs Distance",
            colour="green",
            text=f'Mean RV Diameter: {self.mean_rv_diam:.2f} mm',
            fontsize=self.fontsize
        )
        self.plot_line_graph(
            self.ax[1,3],
            self.slice_positions,
            self.rv_lv_ratios,
            xlabel="Distance along LV long axis (mm)",
            ylabel="RV/LV Diameter Ratio",
            title="RV/LV Diameter Ratio vs Distance",
            colour="orange",
            text=f'Mean RV/LV Ratio: {self.mean_rv_lv_ratio:.2f}',
            fontsize=self.fontsize
        )


    def structure_thickness_analysis(self, endocardial_mask, epicardial_mask, image):
        """ Calculates myocardial wall thicknesses in the SA image space"""

        dist_endo = distance_transform_edt(endocardial_mask==0)
        dist_epi = distance_transform_edt(epicardial_mask==0)

        endo_surface_pts = np.column_stack(np.where(dist_endo == 1))
        epi_surface_pts = np.column_stack(np.where(dist_epi == 1))

        endo_points_mm = np.array([image.TransformIndexToPhysicalPoint((int(p[2]), int(p[1]), int(p[0]))) for p in endo_surface_pts])
        epi_points_mm = np.array([image.TransformIndexToPhysicalPoint((int(p[2]), int(p[1]), int(p[0]))) for p in epi_surface_pts])

        tree_epi = cKDTree(epi_points_mm)

        distances, indices = tree_epi.query(endo_points_mm, k=1)

        # LV centre in physical space
        lv_centre_mm = np.mean(endo_points_mm, axis=0)

        # Radial direction from LV centre
        radial = endo_points_mm - lv_centre_mm
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)

        # Vector from endo → matched epi
        epi_matched = epi_points_mm[indices]
        vec = epi_matched - endo_points_mm

        # Keep only outward matches
        valid = np.sum(vec * radial, axis=1) > 0
        endo_pts_vox = endo_surface_pts[valid]
        endo_points_mm   = endo_points_mm[valid]
        thickness_mm  = distances[valid]

        # we can also get the AHA segment for each point
        aha_segs = self.aha_lv_myo[endo_pts_vox[:,0], endo_pts_vox[:,1], endo_pts_vox[:,2]]

        # for each AHA segment, get mean thickness, 10th percentile, 90th percentile, stddev
        thickness_stats = {}
        for seg in range(1, 18):
            seg_thickness = thickness_mm[aha_segs == seg]
            if len(seg_thickness) == 0:
                continue
            thickness_stats[seg] = {
                "mean_thickness_mm": np.mean(seg_thickness),
                "p10_thickness_mm": np.percentile(seg_thickness, 10),
                "p90_thickness_mm": np.percentile(seg_thickness, 90),
                "stddev_thickness_mm": np.std(seg_thickness)
            }
        # add the overall stats
        overall_thickness = thickness_mm
        thickness_stats["overall"] = {
            "mean_thickness_mm": np.mean(overall_thickness),
            "p10_thickness_mm": np.percentile(overall_thickness, 10),
            "p90_thickness_mm": np.percentile(overall_thickness, 90),
            "stddev_thickness_mm": np.std(overall_thickness)
        }
        return thickness_stats
    

    @staticmethod
    def plot_line_graph(ax, x, y, xlabel, ylabel, title, colour="blue", text="", fontsize=4):
        x_low = min(x) 
        x_high = max(x)
        ax2 = ax.twiny()
        ax2.set_xlim(x_low, x_high)
        ax2.set_xticks([x_low, 0, x_high])
        ax2.set_xticklabels(['Apical', 'Mid LV', 'Basal'], fontsize=fontsize)
        # Move secondary axis to bottom
        ax2.xaxis.set_ticks_position("bottom")
        ax2.xaxis.set_label_position("bottom")
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.spines["bottom"].set_position(("outward", 30))
        ax2.tick_params(axis="x", length=0)
        ax2.tick_params(axis='both', labelsize=fontsize)

        ax.plot(x, y, linestyle='-', color=colour)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize+1)
        ax.tick_params(axis='both', labelsize=fontsize, width=1, length=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.grid(True)
        if text:
            bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="black", alpha=0.5)
            ax.text(0.03, 0.96, f'{text}', color='gold', transform=ax.transAxes,
                    fontsize=fontsize, ha='left', va='top', bbox=bbox_props)


    @staticmethod
    def plot_slice_with_septal_angle(ax, sep_data, fontsize=4):
        septum_edge_1 = sep_data["septum_edge_1"]
        septum_edge_2 = sep_data["septum_edge_2"]
        lv_center = sep_data["lv_center"]
        mid_septum = sep_data["mid_septum"]
        sep_angle = sep_data["septum_angle"]
        lv_width_mm = sep_data["lv_diameter_mm"]
        rv_width_mm = sep_data["rv_diameter_mm"]
        rv_lv_ratio = sep_data["rv_lv_ratio"]
        lv_y_min = sep_data["lv_start_point"][1]
        lv_y_max = sep_data["lv_end_point"][1]
        lv_x_min = sep_data["lv_start_point"][0]
        lv_x_max = sep_data["lv_end_point"][0]
        rv_y_min = sep_data["rv_start_point"][1]
        rv_y_max = sep_data["rv_end_point"][1]
        rv_x_min = sep_data["rv_start_point"][0]
        rv_x_max = sep_data["rv_end_point"][0]
        # plot the septum lines and points
        ax.plot([septum_edge_1[0], mid_septum[0], septum_edge_2[0]], [septum_edge_1[1], mid_septum[1], septum_edge_2[1]], 'aqua')
        ax.plot([lv_center[0], mid_septum[0]], [lv_center[1], mid_septum[1]], ':r', marker='o', markersize=1)
        ax.plot(septum_edge_1[0], septum_edge_1[1], color='red', marker='o', markersize=1)
        ax.plot(septum_edge_2[0], septum_edge_2[1], color='red', marker='o', markersize=1)
        txt = ax.text(sep_data["lv_center"][0]+4, sep_data["lv_center"][1]-4, f'{int(np.round(sep_angle))}°', fontsize=fontsize, color='w')
        txt.set_path_effects([pe.withStroke(linewidth=1, foreground='k')])

        # plot LV and RV diameters
        ax.plot([lv_x_min, lv_x_max], [lv_y_min, lv_y_max], 'blue', linewidth=1)
        ax.plot([rv_x_min, rv_x_max], [rv_y_min, rv_y_max], 'green', linewidth=1)

        # plot RV/LV ratio
        description = f'RV_LV Ratio: {rv_lv_ratio:.2f}\nLV Diameter: {lv_width_mm:.2f} mm\nRV Diameter: {rv_width_mm:.2f} mm'
        bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="black", alpha=0.5)
        ax.text(0.03, 0.96, f'{description}', color='gold', transform=ax.transAxes,
                fontsize=fontsize, ha='left', va='top', bbox=bbox_props)
        

    def calculate_rvlv_diameter_horizontal(self, lv_endo_mask, rv_endo_mask, z_slice):
        """
        Calculate LV and RV diameters and their ratio at the slice specified by z_slice.
        This simply calculates the maximum width of the LV and RV endocardial masks in the x-direction,
        """
        y_max_lv = np.argmax([
            (np.where(lv_endo_mask[y])[0].max() - np.where(lv_endo_mask[y])[0].min())
            if np.any(lv_endo_mask[y]) else 0
            for y in range(lv_endo_mask.shape[0])
        ])
        xs_lv = np.where(lv_endo_mask[y_max_lv])[0]

        y_max_rv = np.argmax([
            (np.where(rv_endo_mask[y])[0].max() - np.where(rv_endo_mask[y])[0].min())
            if np.any(rv_endo_mask[y]) else 0
            for y in range(rv_endo_mask.shape[0])
        ])
        xs_rv = np.where(rv_endo_mask[y_max_rv])[0]

        lv_min = self.img_sa.TransformIndexToPhysicalPoint((int(xs_lv.min()), int(y_max_lv), int(z_slice)))[0]
        lv_max = self.img_sa.TransformIndexToPhysicalPoint((int(xs_lv.max()), int(y_max_lv), int(z_slice)))[0]
        rv_min = self.img_sa.TransformIndexToPhysicalPoint((int(xs_rv.min()), int(y_max_rv), int(z_slice)))[0]
        rv_max = self.img_sa.TransformIndexToPhysicalPoint((int(xs_rv.max()), int(y_max_rv), int(z_slice)))[0]

        lv_width_mm = lv_max - lv_min
        rv_width_mm = rv_max - rv_min
        rv_lv_ratio = rv_width_mm / lv_width_mm

        return {
            "lv_diameter_mm": lv_width_mm,
            "rv_diameter_mm": rv_width_mm,
            "rv_lv_ratio": rv_lv_ratio,
            "lv_start_point": (xs_lv.min(), y_max_lv),
            "lv_end_point": (xs_lv.max(), y_max_lv),
            "rv_start_point": (xs_rv.min(), y_max_rv),
            "rv_end_point": (xs_rv.max(), y_max_rv),
            "z_slice": z_slice
        }


    def calculate_rvlv_diameter_perpendicular_to_septum(self, lv_endo_mask, rv_endo_mask, sep_data, z_slice):
        """
        Calculate LV and RV diameters and their ratio at the slice specified by z_slice.
        This calculates the diameters perpendicular to the septum line.
        """

        # get our septum line points
        septum_edge_1 = sep_data["septum_edge_1"]
        septum_edge_2 = sep_data["septum_edge_2"]
        septum_points = sep_data["septum_points"]

        # compute the septum direction vector (tangent)
        sep_vector = np.array(septum_edge_2) - np.array(septum_edge_1)
        sep_vector = sep_vector / np.linalg.norm(sep_vector)

        # compute the perpendicular vector (normal)
        perp_vector = np.array([-sep_vector[1], sep_vector[0]])

        # for each point along the septum find the intersection with LV and RV masks for a diameter parallel to perp_vector
        lv_diameters = []
        lv_points = []
        for pt in septum_points:
            start_point, end_point = self.ray_mask_intersection(lv_endo_mask, pt, perp_vector)
            if start_point is not None and end_point is not None:
                start_phys = self.img_sa.TransformIndexToPhysicalPoint((int(start_point[0]), int(start_point[1]), int(z_slice)))
                end_phys = self.img_sa.TransformIndexToPhysicalPoint((int(end_point[0]), int(end_point[1]), int(z_slice)))
                diameter = np.linalg.norm(np.array(end_phys) - np.array(start_phys))
                lv_diameters.append(diameter)
                lv_points.append((start_point, end_point))

        rv_diameters = []
        rv_points = []
        for pt in septum_points:
            start_point, end_point = self.ray_mask_intersection(rv_endo_mask, pt, -perp_vector)
            if start_point is not None and end_point is not None:
                start_phys = self.img_sa.TransformIndexToPhysicalPoint((int(start_point[0]), int(start_point[1]), int(z_slice)))
                end_phys = self.img_sa.TransformIndexToPhysicalPoint((int(end_point[0]), int(end_point[1]), int(z_slice)))
                diameter = np.linalg.norm(np.array(end_phys) - np.array(start_phys))
                rv_diameters.append(diameter)
                rv_points.append((start_point, end_point))
        
        lv_diameter_mm = np.max(lv_diameters) if lv_diameters else 0
        rv_diameter_mm = np.max(rv_diameters) if rv_diameters else 0
        rv_lv_ratio = rv_diameter_mm / lv_diameter_mm if lv_diameter_mm != 0 else 0
        # find our max diameter points
        lv_max_idx = np.argmax(lv_diameters) if lv_diameters else None
        rv_max_idx = np.argmax(rv_diameters) if rv_diameters else None
        lv_start_point, lv_end_point = lv_points[lv_max_idx] if lv_max_idx is not None else (None, None)
        rv_start_point, rv_end_point = rv_points[rv_max_idx] if rv_max_idx is not None else (None, None)

        return {
            "lv_diameter_mm": float(lv_diameter_mm),
            "rv_diameter_mm": float(rv_diameter_mm),
            "rv_lv_ratio": float(rv_lv_ratio),
            "lv_start_point": lv_start_point.tolist() if lv_start_point is not None else None,
            "lv_end_point": lv_end_point.tolist() if lv_end_point is not None else None,
            "rv_start_point": rv_start_point.tolist() if rv_start_point is not None else None,
            "rv_end_point": rv_end_point.tolist() if rv_end_point is not None else None,
            "z_slice": z_slice
        }
    

    # https://matplotlib.org/stable/gallery/specialty_plots/leftventricle_bullseye.html
    @staticmethod
    def bullseye_plot(ax, data, segment_stats=None, seg_bold=None, cmap="cividis", norm=None):
        """
        Bullseye representation for the left ventricle.

        Parameters
        ----------
        ax : Axes
        data : list[float]
            The intensity values for each of the 17 segments.
        seg_bold : list[int], optional
            A list with the segments to highlight.
        cmap : colormap, default: "cividis"
            Colormap for the data.
        norm : Normalize or None, optional
            Normalizer for the data.

        Notes
        -----
        This function creates the 17 segment model for the left ventricle according
        to the American Heart Association (AHA) [1]_

        References
        ----------
        .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
            S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
            and M. S. Verani, "Standardized myocardial segmentation and
            nomenclature for tomographic imaging of the heart",
            Circulation, vol. 105, no. 4, pp. 539-542, 2002.
        """
        def text_color_for_background(rgba, threshold=0.5):
            """
            Choose black or white text based on background luminance.

            rgba : tuple (r, g, b, a) in [0, 1]
            """
            r, g, b, _ = rgba

            # Perceptual luminance (ITU-R BT.709)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

            return "black" if luminance > threshold else "white"

        def annotate_segment(ax, theta_c, r_c, text, fontsize=7, color="black"):
            ax.text(
                theta_c,
                r_c,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                clip_on=False
            )

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        data = np.ravel(data)
        if seg_bold is None:
            seg_bold = []
        if norm is None:
            norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

        r = np.linspace(0.2, 1, 4)

        ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
        ax.grid(False)  # Remove grid

        # Fill segments 1-6, 7-12, 13-16, 17
        for start, stop, r_in, r_out in [
                (0, 6, r[2], r[3]),
                (6, 12, r[1], r[2]),
                (12, 16, r[0], r[1]),
                (16, 17, 0, r[0]),
        ]:
            n = stop - start
            dtheta = 2*np.pi / n

            for i in range(n):
                seg_idx = start + i          # 0-based
                seg_id  = seg_idx + 1        # AHA 1–17

                theta0 = i * dtheta + np.pi/2
                theta_c = theta0 + (dtheta / 2)
                r_c = 0.5 * (r_in + r_out)

                # draw the segment
                ax.bar(
                    theta0,
                    r_out - r_in,
                    dtheta,
                    r_in,
                    color=cmap(norm(data[seg_idx]))
                )
                # add segment number
                # ax.text(
                #     theta0,
                #     r_c - 0.05,
                #     f"S{seg_id}",
                #     ha="center",
                #     va="center",
                #     fontsize=6,
                #     alpha=0.5
                # )
                # ---- ADD ANNOTATION HERE ----
                if segment_stats and seg_id in segment_stats:
                    text = segment_stats[seg_id]
                    rgba = cmap(norm(data[seg_idx]))
                    text_colour = text_color_for_background(rgba)
                
                    annotate_segment(
                        ax,
                        theta0,
                        r_c if seg_id != 17 else 0,
                        text,
                        fontsize = 8,
                        color=text_colour
                    )

        # Now, draw the segment borders.  In order for the outer bold borders not
        # to be covered by inner segments, the borders are all drawn separately
        # after the segments have all been filled.  We also disable clipping, which
        # would otherwise affect the outermost segment edges.
        # Draw edges of segments 1-6, 7-12, 13-16.
        for start, stop, r_in, r_out in [
                (0, 6, r[2], r[3]),
                (6, 12, r[1], r[2]),
                (12, 16, r[0], r[1]),
        ]:
            n = stop - start
            dtheta = 2*np.pi / n
            ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
                clip_on=False, color="none", edgecolor="k", linewidth=[
                    4 if i + 1 in seg_bold else 2 for i in range(start, stop)])
        # Draw edge of segment 17 -- here; the edge needs to be drawn differently,
        # using plot().
        ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
                linewidth=(4 if 17 in seg_bold else 2))

        # Add colorbar
        # need to get the plot to add the colorbar
        this_plt = ax.get_figure()

        cbar = this_plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='vertical', fraction=0.02, pad=0.04, location='right')

    @staticmethod
    def ray_mask_intersection(mask, start_point, direction, max_steps=500):
        """
        Cast a ray from start_point in direction (pixel space)
        and return entry and exit points (pixel coordinates).
        """
        direction = direction / np.linalg.norm(direction)

        # check we are going in the correct direction
        cog_mask = np.mean(np.column_stack(np.where(mask)), axis=0)[::-1]  # X,Y
        to_cog = cog_mask - start_point
        if np.dot(direction, to_cog) < 0:
            direction = -direction

        entered = False
        first_idx = None

        prev_x = None
        prev_y = None

        for step in range(max_steps):
            x = int(round(start_point[0] + step * direction[0]))
            y = int(round(start_point[1] + step * direction[1]))

            if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
                return None, None  # Out of bounds

            inside = mask[y, x]

            if inside and not entered:
                entered = True
                first_idx = np.array([x, y])

            elif entered and not inside:
                last_idx = np.array([prev_x, prev_y])
                return first_idx, last_idx
            
            prev_x = x
            prev_y = y

        return None, None  # No intersection found


    @staticmethod
    def mask_to_physical_points(mask, img):
        idx = np.column_stack(np.where(mask))  # Z,Y,X
        pts = np.array([
            img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
            for z,y,x in idx
        ])
        return pts
    

    # https://github.com/alireza-hokmabadi/SegKit/blob/master/segkit/utils/visualization/visualization_sax.py
    @staticmethod
    def calculate_midpoint(points):
        """Calculate the midpoint of a curve (e.g., septum)."""

        if len(points) < 3:
            raise ValueError("Insufficient points to calculate midpoint.")

        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        # Find the index closest to half the total arc length
        idx_mid = np.argmin(np.abs(cumulative_distances - cumulative_distances[-1] / 2))
        return points[idx_mid]
    
    
    # https://github.com/alireza-hokmabadi/SegKit/blob/master/segkit/utils/visualization/visualization_sax.py
    @staticmethod
    def calculate_angle(p1, p2, p3, return_degrees=True):
        """
        Calculate the angle between three points in 2D/3D space.

        Parameters:
            p1, p2, p3: Arrays or lists representing the coordinates of the points.
            return_degrees: Boolean, if True returns angle in degrees, else in radians.

        Returns:
            Angle in degrees (default) or radians.
        """
        # Vectors from p2 to p1 and p2 to p3
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        # Dot product and magnitudes
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # Calculate cosine and ensure numerical stability
        cosine_angle = np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0)
        angle_rad = np.arccos(cosine_angle)

        # Return angle in desired units
        if return_degrees:
            return np.degrees(angle_rad)
        return angle_rad
    

    # https://github.com/alireza-hokmabadi/SegKit/blob/master/segkit/utils/visualization/visualization_sax.py
    @staticmethod
    def order_contour_points(points, threshold=3):
        """
        Orders contour points into a single connected sequence based on proximity.

        Parameters:
            points (np.ndarray): Array of contour points (N x 2).
            threshold (float): Distance threshold to identify disconnections.

        Returns:
            np.ndarray: Ordered contour points as a single continuous array.
        """
        if len(points) <= 2:
            return points  # Nothing to sort for 2 or fewer points

        # Step 1: Identify jumps in contour points based on distance
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        jump_indices = np.where(distances > threshold)[0] + 1
        jump_indices = np.insert(jump_indices, 0, 0)
        jump_indices = np.append(jump_indices, len(points))

        # Split points into disconnected segments
        segments = [points[jump_indices[i]:jump_indices[i+1]] for i in range(len(jump_indices) - 1)]

        # Step 2: Iteratively merge segments into a single connected component
        while len(segments) > 1:
            # Calculate pairwise distances between segment edges
            distances_to_merge = []
            first_segment_edges = [segments[0][0], segments[0][-1]]
            for i in range(1, len(segments)):
                other_segment_edges = [segments[i][0], segments[i][-1]]
                distances_to_merge.append(distance.cdist(first_segment_edges, other_segment_edges))

            # Find the closest segment to merge
            distances_to_merge = np.array(distances_to_merge)
            min_idx = np.unravel_index(np.argmin(distances_to_merge), distances_to_merge.shape)
            min_distance = distances_to_merge[min_idx]

            if min_distance < threshold:
                # Determine merging order
                first_segment = segments[0]
                if min_idx[1] == 0:
                    first_segment = first_segment[::-1]  # Reverse first segment

                second_segment = segments[min_idx[0] + 1]
                if min_idx[2] != 0:
                    second_segment = second_segment[::-1]  # Reverse second segment

                # Merge segments
                merged_segment = np.concatenate((first_segment, second_segment), axis=0)
                segments.append(merged_segment)

                # Remove merged segments from the list
                del segments[min_idx[0] + 1]
                del segments[0]
            else:
                # Keep the largest segment if no further merging is possible
                segment_lengths = [len(seg) for seg in segments]
                segments = [segments[np.argmax(segment_lengths)]]

        return segments[0]


    # https://github.com/alireza-hokmabadi/SegKit/blob/master/segkit/utils/visualization/visualization_sax.py
    def extract_septum(self, endo, epi, rv):
        """Extract the septum region and compute related metrics."""
        # Extract epicardial contour points
        contours, _ = cv2.findContours(cv2.inRange(epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        epi_points = contours[0][:, 0, :]

        # Find the septum by intersecting LV epicardium with RV
        septum_points = []
        dilation_step = 1
        max_dilation_steps = 100  # Limit to prevent infinite loop

        while len(septum_points) == 0:
            # Check if we have a potential infinite loop
            if dilation_step > max_dilation_steps:
                raise ValueError(
                    f"Unable to find an intersection between LV epicardium and RV after {max_dilation_steps} dilation steps."
                )

            # Dilate the RV mask until it intersects with the LV epicardium. Normally, this is fulfilled after just one iteration.
            rv_dilated = cv2.dilate(rv, np.ones((3, 3), dtype=np.uint8), iterations=dilation_step)
            dilation_step += 1
            for y, x in epi_points:
                if rv_dilated[x, y] == 1:
                    septum_points.append(np.array([y, x]))

        # Order septum points
        septum_points = np.array(septum_points)
        ordered_septum = self.order_contour_points(septum_points)

        # Calculate key points of the septum
        mid_septum = self.calculate_midpoint(ordered_septum)
        septum_edge_1 = ordered_septum[0]
        septum_edge_2 = ordered_septum[-1]

        # Calculate the center of the LV cavity
        center_y, center_x = center_of_mass(endo)
        lv_center = np.round(np.array([center_x, center_y])).astype(int)

        # Calculate the septum angle
        angle_1 = self.calculate_angle(septum_edge_1, mid_septum, lv_center, return_degrees=True)
        angle_2 = self.calculate_angle(septum_edge_2, mid_septum, lv_center, return_degrees=True)
        total_septum_angle = angle_1 + angle_2

        return {
            "septum_angle": float(total_septum_angle),
            "septum_points": ordered_septum.tolist(),
            "mid_septum": mid_septum.tolist(),
            "septum_edge_1": septum_edge_1.tolist(),
            "septum_edge_2": septum_edge_2.tolist(),
            "lv_center": lv_center.tolist()
        }
    
    @staticmethod
    def create_aha_masks(
        image_shape,
        centre_point,
        ref_vector,
        location
    ):
        """
        Create angular AHA-style masks for a short-axis slice.

        Parameters
        ----------
        image_shape : (H, W)
        centre_point : (x0, y0)
        ref_vector : (vx, vy), LA -> septum direction
        location : str ("basal", "mid", "apical" or "apex_cap")

        Returns
        -------
        masks : ndarray of shape (num_segments, H, W) where num_segments is 1, 4, or 6 depending on location
        6-segment masks for "basal" and "mid", 4-segment masks for "apical", and 1-segment mask for "apex_cap".

        """

        H, W = image_shape
        x0, y0 = centre_point

        # Normalize reference vector
        ref = np.array(ref_vector, dtype=float)
        ref /= np.linalg.norm(ref)

        # Coordinate grid
        ys, xs = np.mgrid[0:H, 0:W]
        dx = xs - x0
        dy = ys - y0

        # Mask out centre pixel to avoid zero-length vectors
        mag = np.sqrt(dx**2 + dy**2)
        valid = mag > 0

        dx = dx.astype(float)
        dy = dy.astype(float)

        dx /= np.where(valid, mag, 1)
        dy /= np.where(valid, mag, 1)

        # Signed angle between ref and pixel vector
        dot = ref[0]*dx + ref[1]*dy
        cross = ref[0]*dy - ref[1]*dx
        angles = np.arctan2(cross, dot)  # [-pi, pi)

        if location == "basal":
            num_segments = 6
            seg_indices = [5, 4, 3, 2, 1, 6]  # AHA segment numbering
        elif location == "mid":
            num_segments = 6
            seg_indices = [11, 10, 9, 8, 7, 12]  # AHA segment numbering
        elif location == "apical":
            num_segments = 4
            seg_indices = [16, 15, 14, 13]  # AHA segment numbering
        elif location == "apex_cap":
            num_segments = 1
            seg_indices = [17]  # AHA segment numbering

        mask = np.zeros((H, W), dtype=int)
        if num_segments == 1:
            mask[:, :] = 17
            return mask

        if num_segments == 4:
            seg_width = np.pi / 2
            angles_shifted = angles + seg_width / 2
            seg_idx = np.floor(angles_shifted / seg_width).astype(int) % 4

        elif num_segments == 6:
            seg_width = np.pi / 3
            seg_idx = np.floor(angles / seg_width).astype(int) % 6

        else:
            raise ValueError("num_segments must be 1, 4, or 6")

        for s in range(num_segments):
            mask[seg_idx == s] = seg_indices[s]

        return mask
    
if __name__ == "__main__":
    image_path = r"data\1.3.6.1.4.132274.66598776.77819905820063.1301827053.3.2_image.nii.gz"
    label_path = r"data\1.3.6.1.4.132274.66598776.77819905820063.1301827053.3.2_Sharkey_Segmentation.nii.gz"
    output_path = r"data\sax_analysis.png"

    tool = SADiameterTool(image_path, label_path, output_path=output_path)
        