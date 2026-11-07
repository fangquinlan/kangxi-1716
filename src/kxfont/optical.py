"""Bounded, isotropic normalization and bidirectional optical centering.

The multi-scale ink/support and capped-counterspace terms adapt the local
Greek BD experiment to a fixed Han cell. Weights are design choices.
"""
import numpy as np,cv2
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
from skimage.morphology import skeletonize

def measure(mask,pitch):
 y,x=np.nonzero(mask);x=x.astype(float)+.5;y=y.astype(float)+.5
 x0,x1=x.min()-.5,x.max()+.5;y0,y1=y.min()-.5,y.max()+.5
 dt=cv2.distanceTransform(np.pad(mask.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
 sk=skeletonize(mask);sw=2*dt[sk];sw=sw[sw>=2]
 stroke=float(np.median(sw)) if len(sw) else 2.
 cx,cy=x.mean(),y.mean();radius=float(np.sqrt(np.mean((x-cx)**2+(y-cy)**2)))
 complexity=float(len(x)/stroke**2)
 # Optical moments are measured on a capped 96-pixel canvas, not on a
 # resized font outline. Every output point uses one uniform scale only.
 h,w=mask.shape;fac=min(1.,96/max(h,w));a=cv2.resize(mask.astype(np.float32),(round(w*fac),round(h*fac)),interpolation=cv2.INTER_AREA)
 yy,xx=np.indices(a.shape);xx=(xx+.5)/fac;yy=(yy+.5)/fac
 centers=[]
 for sigma in [.018,.04,.08]:
  field=gaussian_filter(a,sigma=max(.5,pitch*fac*sigma),mode='constant',truncate=3)
  # Saturated density reduces the domination of a single thick stroke.
  field=1-np.exp(-field/.35);z=field.sum()
  centers.append([float((field*xx).sum()/z),float((field*yy).sum()/z)])
 # Stable soft facing envelopes along both axes; deep counters are capped.
 counter=[]
 for axis in [0,1]:
  m=mask if axis==0 else mask.T;valid=m.any(1);s=m[valid];mass=s.sum(1).astype(float)
  left=s.argmax(1).astype(float);right=m.shape[1]-s[:,::-1].argmax(1)
  lo=left.min();hi=right.max();left=np.minimum(left,lo+.13*pitch);right=np.maximum(right,hi-.13*pitch)
  ww=np.sqrt(mass);ww/=ww.sum();tau=.04*pitch
  l=-tau*logsumexp(-left/tau,b=ww);r=tau*logsumexp(right/tau,b=ww)
  counter.append(float((l+r)/2))
 sy,sx=np.nonzero(sk)
 return dict(width=w,height=h,bbox=[x0,y0,x1,y1],ink_area=len(x),stroke_px=stroke,
             radius_px=radius,complexity=complexity,ink_center=[cx,cy],
             skeleton_center=[float((sx+.5).mean()),float((sy+.5).mean())],
             field_centers=centers,counter_center=counter,
             components=int(cv2.connectedComponents(mask.astype(np.uint8),8)[0]-1))

def fit_scales(rows,features,config):
 pitch=np.array([r['pitch_px'] for r in rows]);stroke=np.array([f['stroke_px'] for f in features])/pitch
 radius=np.array([f['radius_px'] for f in features])/pitch
 complexity=np.log(np.array([f['complexity'] for f in features]));aspect=np.log(np.array([(f['bbox'][2]-f['bbox'][0])/(f['bbox'][3]-f['bbox'][1]) for f in features]))
 # Robust size prior conditional on complexity and natural aspect. Do not
 # normalize all glyphs to one ink area/bounding box.
 c=complexity-np.median(complexity);asp=np.abs(aspect)
 X=np.column_stack([np.ones(len(rows)),c,c*c,asp,asp*asp,c*asp]);target=np.log(radius)
 weights=np.array([1. if float(r['字形相似度'] or 0)>=.6 else .2 for r in rows]);beta=np.linalg.lstsq(X*weights[:,None],target*weights,rcond=None)[0]
 for _ in range(8):
  resid=target-X@beta;mad=1.4826*np.median(np.abs(resid-np.median(resid)))+.0001
  ww=weights*np.minimum(1,1.5*mad/np.maximum(np.abs(resid),1e-8))
  beta=np.linalg.lstsq(X*np.sqrt(ww[:,None]),target*np.sqrt(ww),rcond=None)[0]
 residual=X@beta-target;sw=np.log(np.median(stroke)/stroke)
 # Eight percent is a trust region, not a claim about perceptual optimality.
 limit=config['scale_correction_limit'];correction=np.exp(np.clip(.18*residual+.12*sw,np.log(1-limit),np.log(1+limit)))
 scales=1000/pitch*correction
 return scales,dict(regression_beta=beta.tolist(),median_stroke_over_pitch=float(np.median(stroke)),weights=dict(radius=.18,stroke=.12),correction_percentiles=np.percentile(correction,[0,1,5,50,95,99,100]).tolist(),note='Engineering proxy; no learned reader preference or uniform-area constraint.')

def place(f,scale,config,curve_bounds=None):
 # Use exact Bezier extrema for the constraints, while keeping all optical
 # features measured on the unchanged original mask.
 x0,y0,x1,y1=curve_bounds if curve_bounds is not None else f['bbox'];box=np.array([(x0+x1)/2,(y0+y1)/2])
 centers=[box,f['ink_center'],f['skeleton_center'],*f['field_centers'],f['counter_center']]
 weights=np.array([.45,.12,.06,.10,.08,.06,.13]);centers=np.array(centers)
 # Exact minimizer of sum w_j ||s(c_j-box)+delta||^2 + prior ||delta||^2.
 delta=-scale*np.average(centers-box,axis=0,weights=weights)
 bound=config['position_limit_em']*1000;delta=np.clip(delta,-bound,bound)
 maxdim=max(x1-x0,y1-y0);safety=False
 if maxdim*scale>930:scale=930/maxdim;safety=True;delta=-scale*np.average(centers-box,axis=0,weights=weights);delta=np.clip(delta,-bound,bound)
 # Half-unit reserve covers CFF coordinate rounding; decoded bounds are
 # checked independently after font compilation.
 xlo=30.5-(500+(x0-box[0])*scale);xhi=969.5-(500+(x1-box[0])*scale)
 ylo=30.5-(500+(y0-box[1])*scale);yhi=969.5-(500+(y1-box[1])*scale)
 delta=np.array([np.clip(delta[0],xlo,xhi),np.clip(delta[1],ylo,yhi)])
 tx=500-scale*box[0]+delta[0];ty=500-scale*box[1]+delta[1]
 before=float(np.sum(weights[:,None]*(scale*(centers-box))**2));after=float(np.sum(weights[:,None]*(scale*(centers-box)+delta)**2))
 return dict(scale=float(scale),tx=float(tx),ty=float(ty),dx=float(delta[0]),dy=float(delta[1]),safety_shrink=safety,objective_before=before,objective_after=after)
