import hashlib
import json
from pathlib import Path
from .geometry import boxes_array, match_boxes


def within_scope(box, protocol):
    x1,y1,x2,y2=box
    roi=protocol['roi_xyxy']
    return (roi[0] <= (x1+x2)/2 <= roi[2] and roi[1] <= (y1+y2)/2 <= roi[3]
            and y2-y1 >= protocol['minimum_visible_height'])


def evaluate(labels, predictions, threshold=0.5):
    frames=labels['frames']
    if not frames:
        raise ValueError('No labelled frames')
    indices=[f['source_frame'] for f in frames]
    if indices!=sorted(set(indices)):
        raise ValueError('Labelled frame indices must be unique and sorted')
    by_frame={p['source_frame']:p for p in predictions}
    if len(by_frame)!=len(predictions):
        raise ValueError('Duplicate prediction frame')
    missing=set(indices)-by_frame.keys()
    if missing:
        raise ValueError(f'Missing prediction records for labelled frames: {sorted(missing)}')
    totals={'tp':0,'fp':0,'fn':0,'id_switches':0,'identity_links':0}
    previous={}
    details=[]
    for frame in frames:
        gt=frame['objects']
        boxes_array([g['box'] for g in gt])
        if len({g['id'] for g in gt})!=len(gt):
            raise ValueError('Duplicate ground-truth identity in a frame')
        if any(not within_scope(g['box'],labels['protocol']) for g in gt):
            raise ValueError('Ground truth outside frozen evaluation scope')
        pred=[p for p in by_frame[frame['source_frame']]['tracks'] if within_scope(p['box'],labels['protocol'])]
        if len({p['track_id'] for p in pred})!=len(pred):
            raise ValueError('Duplicate predicted identity in a frame')
        pairs=match_boxes([g['box'] for g in gt],[p['box'] for p in pred],threshold)
        switches=0
        for gi,pi in pairs:
            ident=gt[gi]['id'];pid=pred[pi]['track_id']
            if ident in previous:
                totals['identity_links']+=1
                switches+=int(previous[ident]!=pid)
            previous[ident]=pid
        tp=len(pairs);fp=len(pred)-tp;fn=len(gt)-tp
        for k,v in [('tp',tp),('fp',fp),('fn',fn),('id_switches',switches)]:totals[k]+=v
        details.append({'source_frame':frame['source_frame'],'gt':len(gt),'predicted':len(pred),'tp':tp,'fp':fp,'fn':fn,'id_switches':switches})
    tp,fp,fn=totals['tp'],totals['fp'],totals['fn']
    return dict(totals,labelled_frames=len(frames),ground_truth_boxes=tp+fn,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                iou_threshold=threshold,per_frame=details,
                scope='Sparse-frame diagnostic; same-author labels; not MOTChallenge MOTA/IDF1 or an independent benchmark.')


def evaluate_files(label_path,prediction_path):
    label_path,prediction_path=Path(label_path),Path(prediction_path)
    frozen=json.loads((label_path.parent/'freeze.json').read_text())
    digest=hashlib.sha256(label_path.read_bytes()).hexdigest()
    if digest!=frozen['labels_sha256']:
        raise ValueError('Frozen annotation checksum mismatch')
    labels=json.loads(label_path.read_text())
    predictions=[json.loads(line) for line in prediction_path.read_text().splitlines()]
    result=evaluate(labels,predictions)
    result.update(labels_sha256=digest,predictions_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest())
    return result


def verify_annotation_source(label_path, source_path):
    label_path,source_path=Path(label_path),Path(source_path)
    frozen=json.loads((label_path.parent/'freeze.json').read_text())
    if hashlib.sha256(label_path.read_bytes()).hexdigest()!=frozen['labels_sha256']:
        raise ValueError('Frozen annotation checksum mismatch')
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=frozen['source_sha256']:
        raise ValueError('Video does not match frozen annotation source')
