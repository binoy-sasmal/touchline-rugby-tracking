import json
import numpy as np
import pytest
from rugby_track.geometry import match_boxes
from rugby_track.tracker import Tracker
from rugby_track.evaluate import evaluate, evaluate_files, verify_annotation_source


def d(x,score=.9):return {'box':[x,0,x+20,40],'score':score}


def test_detection_order_does_not_change_identity_and_velocity_bridges_gap():
    tracker=Tracker(max_age=3)
    first=tracker.update(0,[d(0),d(100)])
    ids=[x['track_id'] for x in first]
    second=tracker.update(1,[d(105),d(5)])
    assert [x['track_id'] for x in second]==ids[::-1]
    assert tracker.update(2,[])==[]  # No invented observations on missed frames.
    third=tracker.update(3,[d(15),d(115)])
    assert [x['track_id'] for x in third]==ids


def test_weak_observation_can_recover_but_never_create_a_track():
    tracker=Tracker()
    assert tracker.update(0,[d(100,.3)])==[]
    ident=tracker.update(1,[d(0)])[0]['track_id']
    result=tracker.update(2,[d(2,.3),d(100,.3)])
    assert len(result)==1 and result[0]['track_id']==ident


def test_expired_track_is_not_resurrected_and_reset_does_not_reuse_id():
    tracker=Tracker(max_age=1)
    first=tracker.update(0,[d(0)])[0]['track_id']
    second=tracker.update(2,[d(0)])[0]['track_id']
    assert second!=first
    tracker.reset()
    assert tracker.update(3,[d(0)])[0]['track_id'] not in [first,second]
    with pytest.raises(ValueError):tracker.update(3,[])


def test_gated_assignment_preserves_two_valid_matches():
    # More total IoU from one match must not displace two allowed matches.
    a=[[0,0,20,20],[10,0,30,20]]
    b=[[0,0,20,20],[-8,0,12,20]]
    assert set(match_boxes(a,b,.3))=={(0,1),(1,0)}


def fixture_labels():
    return {'protocol':{'roi_xyxy':[0,0,200,200],'minimum_visible_height':1},'frames':[
      {'source_frame':0,'objects':[{'id':1,'box':[0,0,20,40]}]},
      {'source_frame':1,'objects':[{'id':1,'box':[2,0,22,40]}]},
      {'source_frame':2,'objects':[{'id':1,'box':[4,0,24,40]}]}]}


def test_metrics_count_false_positive_miss_and_switch_after_gap():
    pred=[{'source_frame':0,'tracks':[dict(d(0),track_id=8),dict(d(100),track_id=9)]},
          {'source_frame':1,'tracks':[]},
          {'source_frame':2,'tracks':[dict(d(4),track_id=10)]}]
    result=evaluate(fixture_labels(),pred)
    assert (result['tp'],result['fp'],result['fn'],result['id_switches'])==(2,1,1,1)
    assert result['precision']==pytest.approx(2/3)
    assert result['recall']==pytest.approx(2/3)
    assert result['identity_links']==1


def test_missing_predictions_do_not_silently_skip_labelled_frames():
    with pytest.raises(ValueError,match='Missing prediction'):
        evaluate(fixture_labels(),[])


def test_mutated_labels_fail_freeze_check(tmp_path):
    label=tmp_path/'labels.json';label.write_text(json.dumps(fixture_labels()))
    (tmp_path/'freeze.json').write_text(json.dumps({'labels_sha256':'wrong'}))
    with pytest.raises(ValueError,match='checksum'):
        evaluate_files(label,tmp_path/'predictions.jsonl')


def test_duplicate_predicted_identity_is_rejected():
    pred=[{'source_frame':i,'tracks':[dict(d(0),track_id=8),dict(d(100),track_id=8)]} for i in range(3)]
    with pytest.raises(ValueError,match='Duplicate predicted'):
        evaluate(fixture_labels(),pred)


def test_nonfinite_or_zero_area_boxes_are_rejected():
    with pytest.raises(ValueError):match_boxes([[0,0,0,2]],[],.5)
    with pytest.raises(ValueError):Tracker().update(0,[{'box':[0,0,np.nan,4],'score':.9}])


def test_labels_cannot_be_reused_on_a_different_video(tmp_path):
    import hashlib
    label=tmp_path/'labels.json';label.write_text(json.dumps(fixture_labels()))
    video=tmp_path/'video.mp4';video.write_bytes(b'different source')
    (tmp_path/'freeze.json').write_text(json.dumps({'labels_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),'source_sha256':'original source hash'}))
    with pytest.raises(ValueError,match='Video does not match'):
        verify_annotation_source(label,video)
