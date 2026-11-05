import pytest
from track1_reference.machine import parse

def test_machine_review_requires_explicit_unknown_schema():
 import json
 a=dict(entities_found=True,description='A hand partially covers the object.',object_visible=True,hand_edge_visibility='partial',object_edge_visibility='hidden',projected_relation='unknown',physical_contact='unknown',uncertainty_reasons=['occlusion'],recommended_action='human_check',evidence='The edge is hidden.')
 assert parse('```json\n'+json.dumps(a)+'\n```')==a
 a['physical_contact']='certain'
 with pytest.raises(ValueError):parse(json.dumps(a))

def test_machine_review_rejects_unstructured_or_missing_evidence():
 with pytest.raises(ValueError):parse('Looks fine')
 with pytest.raises(ValueError):parse('{}')

def test_attention_crop_preserves_original_pixels_and_rejects_bad_boxes():
 import io
 import numpy as np
 from PIL import Image
 from track1_reference.machine_refine import crop
 im=np.arange(100*200*3,dtype=np.uint8).reshape(100,200,3);buf=io.BytesIO();Image.fromarray(im).save(buf,format='PNG')
 raw=buf.getvalue();assert crop(raw,[-1,0,1000,1000]) is None
 assert crop(raw,[0,0,0,1000]) is None
 result,bounds=crop(raw,[250,250,750,750]);x0,y0,x1,y1=bounds
 np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(result))),im[y0:y1,x0:x1])
