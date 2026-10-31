"""Data-driven revised figures, enabled only by complete scientific markers."""
import json
import io
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from nsi_garw.figures import configure_publication_style
from nsi_garw.io import sha256_file
from gen_fig_hpatches_real import SHORT

ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/"results/real";OUTPUT=ROOT/"figures/real"


def save(fig,name,inputs):
    for extension in ("png","pdf","tiff"):
        path=OUTPUT/(name+"."+extension)
        if extension=='tiff':
            buffer=io.BytesIO();fig.savefig(buffer,format='tiff',dpi=600)
            buffer.seek(0)
            with Image.open(buffer) as bitmap:
                bitmap.convert('RGB').save(path,format='TIFF',dpi=(600,600),compression='tiff_lzw')
        else:fig.savefig(path,dpi=600)
    plt.close(fig)
    (OUTPUT/(name+".provenance.json")).write_text(json.dumps({"script_sha256":sha256_file(Path(__file__)),"inputs":{str(p):sha256_file(p) for p in inputs}},indent=2),encoding="utf-8")


def fig1(captions):
    source=RESULTS/"megadepth/pca_complete.json"
    if not source.exists():return
    p=json.loads(source.read_text(encoding="utf-8"));arrays=np.load(RESULTS/"megadepth/pca_model_and_example.npz",allow_pickle=False)
    scores=arrays["example_scores"]/arrays["training_score_scale"][None,:]
    xy=arrays["example_coordinates"];nx,ny=len(np.unique(xy[:,0])),len(np.unique(xy[:,1]))
    if nx*ny!=len(scores):raise ValueError("PCA example does not form a native grid")
    data=pd.read_csv(RESULTS/"megadepth/pca_attribute_correlations.csv")
    columns=["image_slope_x","image_slope_y","image_slope_z","Sobel_x","Sobel_y","Sobel_magnitude"]
    matrix=data.pivot(index="component",columns="attribute",values="test_Pearson_r")[columns].to_numpy()
    fig=plt.figure(figsize=(7,4.2));outer=GridSpec(1,2,figure=fig,width_ratios=[4,2.6],wspace=.25)
    grid=outer[0].subgridspec(2,4,wspace=.15,hspace=.15)
    for i in range(8):
        ax=fig.add_subplot(grid[i//4,i%4]);ax.imshow(scores[:,i].reshape(ny,nx),cmap="RdBu_r",vmin=-2,vmax=2,interpolation="nearest")
        ax.set_title(f"PC{i+1} ({p['explained_variance_ratio'][i]*100:.1f}%)",fontsize=9,pad=3);ax.set_xticks([]);ax.set_yticks([])
    fig.text(.26,.97,'Training-standardized PC score (−2 to 2)',ha='center',fontsize=9)
    ax=fig.add_subplot(outer[1]);im=ax.imshow(matrix,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    ax.set_yticks(range(8),[f"PC{i}" for i in range(1,9)],fontsize=8.5)
    ax.set_xticks(range(6),["Slope x","Slope y","Slope z","Sobel x","Sobel y","Magnitude"],rotation=65,ha="right",fontsize=8.5)
    ax.set_title("Held-out attribute correlation",fontsize=9)
    for i in range(8):
        for j in range(6):ax.text(j,i,f"{matrix[i,j]:.2f}".replace('-0.00','0.00'),ha="center",va="center",fontsize=8,color="white" if abs(matrix[i,j])>.6 else "black")
    fig.colorbar(im,ax=ax,fraction=.05,pad=.04,label="Pearson r")
    fig.subplots_adjust(left=.02,right=.98,top=.92,bottom=.23)
    save(fig,"fig1_megadepth_pca",[source,RESULTS/"megadepth/pca_model_and_example.npz",RESULTS/"megadepth/pca_attribute_correlations.csv"])
    captions["1"]=(f"Fig 1. Exploratory ResNet-50 layer3 PCA on MegaDepth. Eight component maps show one test image selected by a fixed hash before component inspection. "
        f"Colors show component scores in training standard-deviation units (clipped at ±2); percentages indicate explained training variance. "
        f"PCA uses {p['training_samples']:,} training features. The right panel shows descriptive correlations from {p['test_samples']:,} features in {p['test_scenes']} held-out scenes. "
        "Slope channels are camera-unaware log-depth proxies, not surface normals. PCA variance order does not define semantic geometry or appearance subspaces. Cluster-resampling sensitivity is in the source data.")


def fig4(captions):
    marker=RESULTS/"learned_texture_nsi/texture_analysis_complete.json"
    if not marker.exists():return
    texture=pd.read_csv(RESULTS/"learned_texture_nsi/texture_grid_summary.csv")
    sift=pd.read_csv(OUTPUT/"fig4_source.csv")
    fig=plt.figure(figsize=(7,5.6));grid=GridSpec(2,2,figure=fig,height_ratios=[1,1.4],hspace=.56,wspace=.45)
    # Use the audited source column names, never infer percentage units.
    for ax,key,lo,hi,multiplier,label,title in [
        (fig.add_subplot(grid[0,0]),"sequence_NSI_mean","NSI_ci_low","NSI_ci_high",1,"Mean sequence NSI","a  SIFT: surviving-feature NSI"),
        (fig.add_subplot(grid[0,1]),"MMA_at_3","MMA_ci_low","MMA_ci_high",100,"MMA at 3 pixels (%)","b  SIFT: matching with all pairs")]:
        if key not in sift.columns:
            # The complete-grid export supplies exact NSI means; matching
            # columns are mapped below from the original figure source schema.
            raise ValueError(f"Fig4 source schema needs explicit mapping:{key};{list(sift.columns)}")
        x=sift.strength.to_numpy();mean=sift[key].to_numpy()*multiplier
        ax.plot(x,mean,"o-",color="#0072B2",markersize=3);ax.fill_between(x,sift[lo].to_numpy()*multiplier,sift[hi].to_numpy()*multiplier,color="#0072B2",alpha=.15)
        ax.set_title(title,loc="left",fontsize=9);ax.set_xlabel("Texture-suppression strength",fontsize=9);ax.set_ylabel(label,fontsize=9);ax.tick_params(labelsize=8)
        ax.spines[["top","right"]].set_visible(False)
    ax=fig.add_subplot(grid[1,:]);order=list(SHORT)
    matrix=texture.pivot(index="representation",columns="strength",values="NSI").loc[order]
    im=ax.imshow(matrix.to_numpy(),cmap="Blues",vmin=0,vmax=max(.2,float(matrix.to_numpy().max())),aspect="auto")
    ax.set_yticks(range(10),[SHORT[i] for i in order],fontsize=9);ax.set_xticks(range(6),[f"{v:g}" for v in matrix.columns],fontsize=8)
    ax.set_xlabel("Texture-suppression strength",fontsize=9);ax.set_title("c  All ten representations: pooled test NSI",loc="left",fontsize=9)
    for i in range(10):
        for j in range(6):ax.text(j,i,f"{matrix.iloc[i,j]:.3f}",ha="center",va="center",fontsize=8,color="white" if matrix.iloc[i,j]>.6*im.norm.vmax else "black")
    bar=fig.colorbar(im,ax=ax,fraction=.025,pad=.02);bar.set_label('Pooled NSI',fontsize=9);bar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.2,right=.97,top=.94,bottom=.09)
    save(fig,"fig4_sift_texture",[marker,RESULTS/"learned_texture_nsi/texture_grid_summary.csv",OUTPUT/"fig4_source.csv"])
    captions["4"]=("Fig 4. Target-only texture suppression on HPatches. Panels a and b show SIFT sequence-mean NSI and matching accuracy with 95% sequence-bootstrap intervals. "
        "Panel c shows pooled test NSI for ten representations and six strengths. Clean-image attributes remain fixed; probes are refitted per condition under the same frozen 81/18/17 sequence labels. "
        "NSI is conditional on surviving aligned native features; matching retains all pairs. At strength one, DeDoDe loses one training sequence, without replacement or reassignment. "
        "Changing locations, coverage, and fitting support prevent a fixed-keypoint interpretation. The complete source table reports coverage and condition-specific uncertainty.")


def fig6(captions):
    marker=RESULTS/"eth3d_garw/analysis_complete.json";tolerance_marker=RESULTS/"eth3d_garw/tolerance_analysis_complete.json"
    if not marker.exists() or not tolerance_marker.exists():return
    hp=pd.read_csv(RESULTS/"garw_hpatches/loftr_outdoor_coarse_1024/non_oracle/garw_per_scene.csv")
    eth=pd.read_csv(RESULTS/"eth3d_garw/per_scene_garw.csv")
    tolerance=pd.read_csv(RESULTS/"eth3d_garw/tolerance_summary.csv")
    fig,axes=plt.subplots(2,2,figsize=(7,5.8));left,right=axes[0]
    methods=["descriptor_baseline","geometry","appearance","fixed_weight","garw_lambda_zero","garw"]
    labels=["Raw descriptor","Geometry","Appearance","Fixed gate","Gate, λ = 0","Full GARW"]
    rng=np.random.default_rng(20261023)
    hp_indices=rng.integers(0,17,size=(10000,17))
    eth_indices=rng.integers(0,5,size=(10000,5))
    for i,method in enumerate(methods):
        values=hp.loc[hp.method==method].sort_values("scene").top1.to_numpy()*100
        lo,hi=np.quantile(values[hp_indices].mean(axis=1),[.025,.975]);mean=values.mean()
        left.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt="o",color="#D55E00" if method=="garw" else "#0072B2",capsize=2,markersize=3)
    left.set_yticks(range(6),labels,fontsize=9);left.invert_yaxis();left.set_xlabel("Top-1 correct (%)",fontsize=9);left.set_title("a  HPatches: LoFTR encoder",loc="left",fontsize=9)
    pivot=hp.pivot(index="scene",columns="method",values="top1").sort_index()
    right.axhline(0,linestyle="--",color="#777777",linewidth=.7);right.scatter(np.arange(1,18),100*(pivot.garw-pivot.descriptor_baseline),s=15,color="#D55E00")
    right.set_xlabel("Held-out sequence index",fontsize=9);right.set_ylabel("GARW − raw (pp)",fontsize=9);right.set_title("b  HPatches: paired differences",loc="left",fontsize=9)
    left,right=axes[1]
    for i,(identifier,label) in enumerate([("loftr_outdoor_coarse_1024","LoFTR"),("roma_outdoor_scale16_560","RoMa")]):
        for method,offset,color in [("descriptor_baseline",-.12,"#0072B2"),("garw",.12,"#D55E00")]:
            values=eth.loc[(eth.representation==identifier)&(eth.method==method)].sort_values("scene").top1.to_numpy()*100
            lo,hi=np.quantile(values[eth_indices].mean(axis=1),[.025,.975]);mean=values.mean()
            left.errorbar(i+offset,mean,yerr=[[mean-lo],[hi-mean]],fmt="o",color=color,capsize=2,markersize=3,label="Raw" if method=="descriptor_baseline" else "GARW")
    left.set_xticks([0,1],["LoFTR","RoMa"],fontsize=9);left.set_ylabel("Top-1 correct (%)",fontsize=9);left.set_title("c  ETH3D: strict 3-pixel evaluation",loc="left",fontsize=9)
    handles,legend=left.get_legend_handles_labels();left.legend(handles[:2],legend[:2],frameon=False,fontsize=8)
    for identifier,color,label in [("loftr_outdoor_coarse_1024","#0072B2","LoFTR"),("roma_outdoor_scale16_560","#D55E00","RoMa")]:
        for method,symbol in [("descriptor_baseline","o"),("garw","s")]:
            block=tolerance.loc[(tolerance.representation==identifier)&(tolerance.method==method)].sort_values("tolerance_original_pixels")
            right.plot(block.tolerance_original_pixels,block.top1*100,marker=symbol,color=color,markersize=3,label=label+(" raw" if method=="descriptor_baseline" else " GARW"))
    right.set_xlabel("Tolerance (original pixels)",fontsize=9);right.set_ylabel("Top-1 correct (%)",fontsize=9);right.set_title("d  ETH3D: tolerance sensitivity",loc="left",fontsize=9);right.legend(frameon=False,fontsize=8)
    for ax in axes.ravel():ax.spines[["top","right"]].set_visible(False);ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.2,right=.98,bottom=.1,top=.94,hspace=.55,wspace=.47)
    save(fig,"fig6_hpatches_garw",[marker,tolerance_marker,RESULTS/"garw_hpatches/loftr_outdoor_coarse_1024/non_oracle/garw_per_scene.csv",RESULTS/"eth3d_garw/per_scene_garw.csv",RESULTS/"eth3d_garw/tolerance_summary.csv"])
    captions["6"]=("Fig 6. Non-oracle encoder-feature candidate reranking. Panels a and b show HPatches LoFTR results and paired GARW-minus-raw differences across 17 test sequences. "
        "Panel c shows two ETH3D encoders on five test scenes at 3-pixel forward reprojection tolerance. Error bars are 95% group-bootstrap intervals with fixed calibration. "
        "ETH3D has sparse correct-candidate coverage at this strict tolerance. Panel d sweeps the tolerance after observing that limitation, without changing candidates or calibrated parameters. "
        "This sensitivity is exploratory. Correct candidates are never injected. These candidate tests do not evaluate the complete paired-matcher heads.")


def fig2(captions):
    marker=RESULTS/"megadepth/analysis_complete.json"
    if not marker.exists():return
    hp=pd.read_csv(RESULTS/"hpatches_nsi_summary.csv").set_index("experiment_id")
    eth=pd.read_csv(RESULTS/"eth3d_nsi_summary.csv").set_index("experiment_id")
    md=pd.read_csv(RESULTS/"megadepth/nsi_summary.csv").set_index("representation")
    order=list(SHORT)
    frames=[("HPatches",hp.loc[order],"group_NSI_mean","group_NSI_CI_low","group_NSI_CI_high",17),
            ("ETH3D",eth.loc[order],"group_NSI_mean","group_NSI_CI_low","group_NSI_CI_high",5),
            ("MegaDepth",md.loc[order],"scene_NSI_mean","scene_NSI_CI_low","scene_NSI_CI_high",int(md.loc[order[0],"test_scenes"]))]
    fig,axes=plt.subplots(2,3,figsize=(7,6.1),gridspec_kw={"height_ratios":[1,1]},sharey="row")
    source_rows=[]
    for j,(name,frame,mean,low,high,n) in enumerate(frames):
        top,bottom=axes[:,j]
        matrix=frame[["A_accessibility","B_separation","C_information_purity","NSI"]].to_numpy()
        image=top.imshow(matrix,cmap="Blues",vmin=0,vmax=1,aspect="auto")
        top.set_title(f"{chr(97+j)}  {name}: pooled",loc="left",fontsize=9)
        top.set_xticks(range(4),["A","B","C","NSI"],fontsize=9)
        top.set_yticks(range(10),[SHORT[i] for i in order],fontsize=9)
        for r in range(10):
            for c in range(4):top.text(c,r,f"{matrix[r,c]:.2f}",ha="center",va="center",fontsize=8,color="white" if matrix[r,c]>.6 else "black")
        values=frame[mean].to_numpy();lo=frame[low].to_numpy();hi=frame[high].to_numpy()
        bottom.errorbar(values,np.arange(10),xerr=[values-lo,hi-values],fmt="o",color="#0072B2",capsize=2,markersize=3)
        bottom.set_yticks(range(10),[SHORT[i] for i in order],fontsize=9);bottom.set_ylim(9.5,-.5)
        bottom.set_xlabel("Group-mean NSI (95% CI)",fontsize=8)
        bottom.set_title(f"{chr(100+j)}  {n} test groups",loc="left",fontsize=9)
        bottom.tick_params(axis="x",labelsize=8);bottom.locator_params(axis="x",nbins=4)
        bottom.spines[["top","right"]].set_visible(False)
        source_rows.extend(frame.reset_index().assign(dataset=name).to_dict("records"))
    fig.subplots_adjust(left=.22,right=.98,bottom=.09,top=.94,hspace=.75,wspace=.2)
    color_axis=fig.add_axes([.34,.505,.5,.018])
    bar=fig.colorbar(image,cax=color_axis,orientation="horizontal")
    bar.set_label("Component value (0–1)",fontsize=8);bar.ax.tick_params(labelsize=8)
    pd.DataFrame(source_rows).to_csv(OUTPUT/"fig2_three_dataset_source.csv",index=False)
    save(fig,"fig2_hpatches_nsi",[RESULTS/"hpatches_nsi_summary.csv",RESULTS/"eth3d_nsi_summary.csv",RESULTS/"megadepth/nsi_summary.csv",marker])
    captions["2"]=("Fig 2. Ten representation diagnostics under three dataset-specific protocols. Top panels show pooled-test A, B, C, and NSI. "
        "Bottom panels show equal-weight group means and 95% group-bootstrap intervals with fixed fitted probes. "
        f"Test groups comprise 17 HPatches sequences, five ETH3D scenes, and {frames[2][-1]} MegaDepth scenes. "
        "HPatches uses scalar homography/gradient proxies; ETH3D uses camera-frame laser normals and three Sobel channels. MegaDepth uses camera-unaware depth-slope and Sobel vectors. "
        "Absolute values are not measurement-invariant. Coverage follows native sampling and valid attributes. Intervals exclude probe-fitting uncertainty.")


def fig3(captions):
    marker=RESULTS/"megadepth/analysis_complete.json"
    if not marker.exists():return
    hp=pd.read_csv(OUTPUT/"fig3_source.csv")
    md=pd.read_csv(RESULTS/"megadepth/nsi_summary.csv").set_index("representation")
    resnet=["maxpool","layer1","layer2","layer3","layer4"];dino=[2,5,8,11]
    fig,axes=plt.subplots(2,2,figsize=(7,5.2));rows=[]
    for row,dataset in enumerate(("HPatches","MegaDepth")):
        for column,model in enumerate(("ResNet-50","DINOv2-S/14")):
            ax=axes[row,column]
            stages=resnet if column==0 else dino
            if dataset=="HPatches":
                block=hp.loc[hp.model==model]
                if len(block)!=len(stages):raise ValueError("Incomplete HPatches stage source")
                pooled=block.pooled_NSI.to_numpy();mean=block.sequence_NSI_mean.to_numpy();lo=block.ci_low.to_numpy();hi=block.ci_high.to_numpy()
            else:
                names=["resnet50_"+s for s in stages] if column==0 else [f"dinov2_vits14_block{s}_518" for s in stages]
                block=md.loc[names]
                pooled=block.NSI.to_numpy();mean=block.scene_NSI_mean.to_numpy();lo=block.scene_NSI_CI_low.to_numpy();hi=block.scene_NSI_CI_high.to_numpy()
            x=np.arange(len(stages))
            ax.plot(x,pooled,"s-",color="#333333",markersize=3,label="Pooled NSI")
            ax.errorbar(x,mean,yerr=[mean-lo,hi-mean],fmt="o-",color="#0072B2",capsize=2,markersize=3,label="Group mean, 95% CI")
            ax.set_xticks(x,["pool" if s=="maxpool" else str(s) for s in stages],fontsize=8)
            ax.set_title(f"{chr(97+row*2+column)}  {dataset}: {model}",loc="left",fontsize=9)
            ax.set_ylabel("NSI",fontsize=9);ax.set_xlabel("Stage" if column==0 else "Block (zero-based)",fontsize=9)
            ax.spines[["top","right"]].set_visible(False);ax.tick_params(labelsize=8);ax.set_ylim(bottom=0)
            if row==0 and column==0:ax.legend(frameon=False,fontsize=8)
            for index,stage in enumerate(stages):rows.append({"dataset":dataset,"model":model,"stage":stage,"pooled_NSI":pooled[index],"group_NSI_mean":mean[index],"ci_low":lo[index],"ci_high":hi[index]})
    fig.subplots_adjust(left=.1,right=.98,top=.94,bottom=.1,wspace=.3,hspace=.55)
    pd.DataFrame(rows).to_csv(OUTPUT/"fig3_two_dataset_source.csv",index=False)
    save(fig,"fig3_hpatches_layerwise",[OUTPUT/"fig3_source.csv",RESULTS/"megadepth/nsi_summary.csv",marker])
    captions["3"]=("Fig 3. ResNet-50 and DINOv2 stage diagnostics on HPatches and MegaDepth. Black squares show pooled test NSI. "
        "Blue circles and error bars show equal-weight test-group means with 95% group-bootstrap intervals and fixed probes. "
        "ResNet uses a 224-pixel center crop; DINO uses a 518-pixel input and zero-based block indices. "
        "The A, B, and C components and stage-specific coverage are provided in the source data. Attribute constructions differ across datasets. "
        "These descriptive test curves do not select a deployment layer or establish a universal peak.")


def fig5(captions):
    source=RESULTS/"hpatches_complete_associations/analysis_complete.json"
    if not source.exists():return
    p=json.loads(source.read_text(encoding="utf-8"));rows=p["rows"]
    order=["SIFT","ResNet-50 layer3","DINOv2 block11","LoFTR native","RoMa native"]
    fig,axes=plt.subplots(1,2,figsize=(7,3.2),sharey=True)
    for index,(target,title) in enumerate([("cleanMMA3","a  Clean matching accuracy"),("paired_texture_minus_clean_MMA3","b  Texture-induced accuracy change")]):
        ax=axes[index]
        for i,label in enumerate(order):
            r=next(r for r in rows if r["representation"]==label and r["target"]==target)
            value=r["spearman_rho"];ci=r["confidence_interval"]
            if value is None:ax.text(0,i,"Not estimable",ha="center",fontsize=7);continue
            if ci is not None:ax.plot(ci,[i,i],color="#0072B2",lw=1.2)
            ax.plot(value,i,"o",color="#0072B2",markersize=4)
            ax.text(.98,i-.16,f"q = {r['BH_q_15_attempted_associations']:.3f}",ha="right",va="bottom",fontsize=8)
        ax.axvline(0,color="#777777",ls=":",lw=.7);ax.set_xlim(-1.05,1.05)
        ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks(range(5),order,fontsize=9)
        ax.set_ylim(4.6,-.8);ax.set_xlabel("Spearman rho\n95% sequence-bootstrap CI",fontsize=9)
        ax.set_title(title,loc="left",fontsize=9);ax.tick_params(labelsize=8);ax.spines[["top","right"]].set_visible(False)
    fig.subplots_adjust(left=.2,right=.98,top=.86,bottom=.18,wspace=.17)
    save(fig,"fig5_sift_association",[source,RESULTS/"hpatches_complete_associations/association_summary.csv"])
    captions["5"]=("Fig 5. Exploratory associations on the same 17 held-out HPatches sequences. Points and lines show Spearman rho and 95% sequence-bootstrap intervals for clean NSI versus clean MMA@3 (a) or the paired texture-minus-clean MMA@3 change (b; texture strength 0.8). "
        "The change is signed: more negative values indicate greater loss. Each sequence averages five pairs at 3-pixel RMS symmetric homography error. "
        "SIFT, ResNet, and DINO use cosine mutual-nearest matching; LoFTR and RoMa use complete native matchers with different candidate sets. "
        "ResNet/DINO auxiliary matching uses uniform CPU FP32; pinned feature definitions are retained. "
        "Intervals use 10,000 sequence resamples and p values use 1,000 permutations. BH q covers all 15 attempted tests across five methods and three targets, including corrupted accuracy in the source data. "
        "This family was expanded after earlier outcomes; these descriptive associations do not establish predictive validation or causation.")


def main():
    configure_publication_style();captions={}
    fig1(captions);fig2(captions);fig3(captions);fig4(captions);fig5(captions);fig6(captions)
    path=OUTPUT/"extension_captions.json"
    old=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(captions);path.write_text(json.dumps(old,indent=2),encoding="utf-8")
    print("Updated completed figure extensions:"+",".join(captions),flush=True)


if __name__=="__main__":main()
