"""Regression: repeated detector indices in separate images keep distinct IDs."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from nsi_garw.adapters import LocalFeatures
from nsi_garw.io import load_feature_npz,sha256_file
import json


def test_export_preserves_image_identity_with_repeated_feature_indices(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/run_megadepth_revision.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('md_identity_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bulk=tmp_path/'bulk';audit=tmp_path/'audit';audit.mkdir()
    exports=tmp_path/'exports';exports.mkdir();results=tmp_path/'results';results.mkdir()
    for name,value in [('DATA',bulk),('AUDIT',audit),('EXPORTS',exports),('FOLDER',results)]:monkeypatch.setattr(module,name,value)
    protocol=audit/'protocol.json';protocol.write_text('{}')
    cohort=audit/'revision_image_cohort.csv';cohort.write_text('fixture only;never manuscript evidence')
    rows=[]
    for scene,split in [('0000','train'),('0001','validation'),('0002','test')]:
        for image_number in (0,1):
            root=bulk/'datasets/MegaDepth/MegaDepth_v1'/scene/f'dense{image_number}'
            root.mkdir(parents=True)
            image=root/'same_photo.jpg';depth=root/'same_photo.h5'
            Image.fromarray(np.full((8,8,3),70+image_number,dtype=np.uint8)).save(image)
            with h5py.File(depth,'w') as file:file.create_dataset('depth',data=np.ones((8,8),dtype=np.float32))
            rows.append({'scene':scene,'split':split,'image':str(image),'depth':str(depth),'image_sha256':sha256_file(image),'depth_sha256':sha256_file(depth)})
    class FixtureAdapter:
        def extract(self,image):
            return LocalFeatures(np.asarray([[2,2],[3,3],[4,4],[5,5]],dtype=float),np.ones((4,5),dtype=np.float32),np.ones(4),{'fixture':'not scientific evidence'})
    module.export('fixture',FixtureAdapter(),pd.DataFrame(rows),protocol)
    data=load_feature_npz(exports/'fixture_unfiltered.npz')
    assert len(data.sample_ids)==24
    assert len(np.unique(data.sample_ids))==24
    for row in rows:
        expected=f"{row['scene']}:{module.image_key(row['depth'])}:"
        assert sum(str(value).startswith(expected) for value in data.sample_ids)==4


def test_existing_windows_cohort_is_read_identically_by_multiple_workers(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/run_megadepth_revision.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('md_newline_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audit=tmp_path/'audit';audit.mkdir();weights=tmp_path/'fixture.pth';weights.write_bytes(b'fixture, not a model')
    monkeypatch.setattr(module,'AUDIT',audit);monkeypatch.setattr(module,'RESNET_WEIGHTS',weights)
    (audit/'cohort_preparation_protocol_v2.json').write_text('{}')
    frame=pd.DataFrame([{'scene':scene,'image':f'/images/{scene}/a.jpg','depth':f'/depths/{scene}/a.h5'} for scene in ('0000','0001','0002')])
    source=audit/'source.csv';frame.to_csv(source,index=False,lineterminator='\r\n')
    (audit/'cohort_complete.json').write_text(json.dumps({'status':'PASS','cohort_csv':str(source),'cohort_sha256':sha256_file(source)}))
    # Existing frozen Windows file is left byte-for-byte intact by every reader.
    frozen=audit/'revision_image_cohort.csv';frame.to_csv(frozen,index=False,lineterminator='\r\n')
    before=frozen.read_bytes()
    for _ in range(3):
        _,actual=module.freeze_and_cohort(False)
        assert actual.scene.tolist()==['0000','0001','0002']
        assert frozen.read_bytes()==before
