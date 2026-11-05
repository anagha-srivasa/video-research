"""Retain source polygons and diagnose them before raster geometry extraction."""
import math
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely import make_valid
from .common import dumps

POLICY='GEOS_linework_for_invalid; polygon_union; holes_subtracted; round_vertices; clip_to_annotation_grid'

def polygon_parts(g):
    if g.geom_type=='Polygon':yield g
    elif hasattr(g,'geoms'):
        for child in g.geoms:yield from polygon_parts(child)

def _raster(segments,width,height):
    flags=[];polys=[]
    repair=None
    def fail(code):return None,0,0,sorted(set(flags+[code])),repair
    if not isinstance(segments,list) or not segments:return fail('empty_segments')
    for ring in segments:
        try:a=np.asarray(ring,dtype=np.float64)
        except (TypeError,ValueError):return fail('malformed_polygon')
        if a.ndim!=2 or a.shape[1]!=2 or len(a)<3:return fail('degenerate_polygon')
        if not np.isfinite(a).all():return fail('nonfinite_coordinates')
        if (a[:,0]<-width).any() or (a[:,0]>2*width).any() or (a[:,1]<-height).any() or (a[:,1]>2*height).any():return fail('extreme_out_of_bounds_coordinates')
        if len(np.unique(a,axis=0))<3:return fail('too_few_distinct_vertices')
        if len(np.unique(a,axis=0))<len(a):flags.append('repeated_vertices')
        p=Polygon(a)
        if not p.is_valid:
            flags.append('source_polygon_invalid');repair='self_intersection_make_valid_linework'
            p=make_valid(p,method='linework')
            if p.geom_type=='GeometryCollection':flags.append('repair_contains_non_area_fragments')
        parts=[part for part in polygon_parts(p) if part.area>0]
        if not parts:return fail('no_area_after_geometry_validation')
        if (a[:,0]<0).any() or (a[:,0]>=width).any() or (a[:,1]<0).any() or (a[:,1]>=height).any():flags.append('bounds_clipped_for_rasterisation')
        polys.extend(parts)
    allpts=np.concatenate([np.asarray(p.exterior.coords) for p in polys])
    x0=max(0,int(np.floor(allpts[:,0].min()))-1);y0=max(0,int(np.floor(allpts[:,1].min()))-1)
    x1=min(width,int(np.ceil(allpts[:,0].max()))+2);y1=min(height,int(np.ceil(allpts[:,1].max()))+2)
    if x1<=x0 or y1<=y0:return fail('outside_image')
    mask=np.zeros((y1-y0,x1-x0),np.uint8)
    for p in polys:
        part=np.zeros_like(mask)
        cv2.fillPoly(part,[np.rint(np.asarray(p.exterior.coords)-[x0,y0]).astype(np.int32)],1)
        for hole in p.interiors:cv2.fillPoly(part,[np.rint(np.asarray(hole.coords)-[x0,y0]).astype(np.int32)],0)
        mask|=part
    if not mask.any():return fail('empty_raster')
    return mask,x0,y0,sorted(set(flags)),repair

def geometry(segments,width,height):
    mask,x0,y0,flags,repair=_raster(segments,width,height)
    result=dict(bbox_xmin=None,bbox_ymin=None,bbox_xmax=None,bbox_ymax=None,area_px=None,
                centroid_x=None,centroid_y=None,perimeter_px=None,component_count=None,hole_count=None,
                contours_json='[]',geometry_status='quarantined',repair_code=repair,rasterisation_policy=POLICY,quality_flags=flags)
    if mask is None:return result
    area=int(mask.sum())
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rings=[];components=0;holes=0;perimeter=0.0
    for i,c in enumerate(contours):
        depth=0;parent=int(hierarchy[0][i][3]);p=parent
        while p!=-1:depth+=1;p=int(hierarchy[0][p][3])
        role='outer' if depth%2==0 else 'inner'
        if role=='outer':components+=1
        else:holes+=1
        points=c[:,0,:]+[x0,y0]
        length=float(cv2.arcLength(c,True));perimeter+=length
        signed=float(cv2.contourArea(c,oriented=True))
        vertices=points.astype(np.float64)
        before=vertices-np.roll(vertices,1,axis=0);after=np.roll(vertices,-1,axis=0)-vertices
        lengths=np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1)
        cosine=np.divide(np.sum(before*after,axis=1),lengths,out=np.ones(len(vertices)),where=lengths>0)
        turns=np.arccos(np.clip(cosine,-1,1));local_length=(np.linalg.norm(before,axis=1)+np.linalg.norm(after,axis=1))/2
        curvature=np.divide(turns,local_length,out=np.zeros(len(vertices)),where=local_length>0)
        rings.append(dict(role=role,parent_ring=parent,depth=depth,points=points.tolist(),arc_length=length,
                          signed_area=signed,sample_kind='raster_boundary_vertices',
                          curvature_abs_mean_per_px=float(curvature.mean()) if len(curvature) else None,
                          curvature_abs_max_per_px=float(curvature.max()) if len(curvature) else None,
                          curvature_status='pixel_boundary_diagnostic_not_smooth_curve_estimate'))
    ys,xs=np.nonzero(mask)
    result.update(bbox_xmin=float(xs.min()+x0),bbox_ymin=float(ys.min()+y0),bbox_xmax=float(xs.max()+x0+1),
                  bbox_ymax=float(ys.max()+y0+1),area_px=float(area),centroid_x=float(xs.mean()+x0),
                  centroid_y=float(ys.mean()+y0),perimeter_px=perimeter,component_count=components,hole_count=holes,
                  contours_json=dumps(rings),geometry_status='repaired_for_rasterisation' if repair else 'clipped_for_rasterisation' if 'bounds_clipped_for_rasterisation' in flags else 'valid_with_diagnostics' if flags else 'valid',quality_flags=sorted(set(flags)))
    return result

def rasterize(segments,width,height):
    roi,x0,y0,flags,repair=_raster(segments,width,height)
    if roi is None:raise ValueError('Quarantined mask: '+str(flags))
    mask=np.zeros((height,width),np.uint8)
    mask[y0:y0+roi.shape[0],x0:x0+roi.shape[1]]=roi
    return mask
