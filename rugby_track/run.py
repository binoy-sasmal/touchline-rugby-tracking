import argparse
import hashlib
import html
import json
import platform
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
from .detector import Detector
from .tracker import Tracker
from .evaluate import evaluate_files, verify_annotation_source


ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def distribution(values):
    return {'median_ms':float(np.median(values)*1000),'p95_ms':float(np.percentile(values,95)*1000)}


def make_report(out,stats,metrics):
    def pct(value):return 'n/a' if value is None else f'{value*100:.1f}%'
    rows=''.join('<tr>'+''.join(f'<td>{row[k]}</td>' for k in ['source_frame','gt','predicted','tp','fp','fn','id_switches'])+'</tr>' for row in metrics['per_frame'])
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Touchline | Rugby tracking baseline</title>
<style>body{{margin:0;background:#101b23;color:#e6edf3;font:16px/1.65 system-ui}}main{{max-width:1100px;margin:auto;padding:40px 24px}}small,.muted{{color:#9baebc}}h1{{font-size:44px;line-height:1.15;margin:12px 0}}h2{{margin-top:40px}}.tag{{color:#62d3b0;text-transform:uppercase;letter-spacing:3px;font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:26px 0}}.card{{background:#1b2b37;padding:20px;border-radius:10px}}strong{{font-size:28px;display:block}}video,img{{width:100%;border-radius:10px}}.table-scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;text-align:left;font-variant-numeric:tabular-nums}}td,th{{padding:9px;border-bottom:1px solid #304551}}a{{color:#62d3b0}}code{{color:#bce4ed}}.note{{border-left:3px solid #62d3b0;padding-left:20px}}footer{{margin-top:45px;color:#9baebc;font-size:13px}}</style>
<main><div class="tag">Computer vision / video engineering</div><h1>Touchline</h1><p class="muted">An inspectable rugby person-tracking baseline. Pretrained detection, persistent track labels, and measured failure modes.</p>
<video controls loop muted playsinline poster="preview.jpg"><source src="annotated.mp4" type="video/mp4"></video>
<div class="cards"><div class="card">Detection precision<strong>{pct(metrics['precision'])}</strong><small>IoU ≥ 0.5, scoped labels</small></div><div class="card">Detection recall<strong>{pct(metrics['recall'])}</strong><small>{metrics['ground_truth_boxes']} boxes / {metrics['labelled_frames']} labelled frames</small></div><div class="card">ID switches<strong>{metrics['id_switches']}</strong><small>{metrics['identity_links']} observed identity links</small></div><div class="card">CPU detector median<strong>{stats['detector']['median_ms']:.0f} ms</strong><small>p95 {stats['detector']['p95_ms']:.0f} ms; warm-up excluded</small></div></div>
<p class="note">Diagnostic evidence, not a benchmark. Annotations were drawn by Codex through visual inspection before inference and have not been independently human reviewed. Eight sparse frames from one short touch-rugby shot cannot establish whole-match accuracy or robust identity tracking.</p>
<h2>What runs</h2><p>YOLOX-S ONNX person detections → two-pass confidence association with constant-velocity box prediction → observed track labels and JSONL export. The tracker is a small baseline, without appearance embeddings, camera compensation or player names.</p>
<h2>Where it fails</h2><p>Motion blur and overlap can hide people or merge boxes. Camera panning changes apparent velocity. A high-confidence threshold can delay a new track; low-confidence observations may recover existing tracks but cannot create one. The labels use visible body extent, while the detector may infer a fuller body, which can lower IoU. IDs have meaning only within this continuous shot.</p>
<h2>Frame-level audit</h2><div class="table-scroll"><table><tr><th>Frame</th><th>Labels</th><th>Predictions</th><th>TP</th><th>FP</th><th>FN</th><th>ID switches</th></tr>{rows}</table></div>
<p>Evaluation counts people whose box centre is inside x=30..610, y=30..320 and whose visible box height is at least 40 px. Includes officials. Smaller/distant and edge-centred people are outside this declared scope. Detection metrics use raw detector outputs; identity switches use tracked matches only.</p>
<h2>Timing boundaries</h2><p>{stats['frames']} consecutive frames, {stats['source_fps']:.0f} fps source. Model load {stats['model_load_ms']:.0f} ms; first-frame warm-up {stats['warmup_ms']:.0f} ms. Median tracker {stats['tracker']['median_ms']:.2f} ms. Measured loop throughput {stats['loop_fps']:.1f} fps includes decode, detector, tracker, rendering and intermediate encoding; excludes model load, warm-up and the separate final H.264 export ({stats["h264_transcode_s"]:.2f} s). This is not a real-time or GPU claim.</p>
<h2>Review without running code</h2><p><a href="annotation-audit.jpg">View frozen annotation audit</a> · <a href="metrics.json">Metrics JSON</a> · <a href="run.json">Run provenance</a> · <a href="predictions.jsonl">Detections and tracks</a></p>
<footer>Footage: The University of Auckland, “2017 Interfaculty Touch Rugby Tournament”, <a href="https://creativecommons.org/licenses/by/2.5/">CC BY 2.5</a>, <a href="https://commons.wikimedia.org/wiki/File:2017-Interfaculty-Touch-Rugby-Tournament_1080p-Nueva_Zelanda.webm">source/licence record</a>. Changes: cropped time interval, no audio, tracking overlays and resizing. No endorsement or affiliation with Athlaite. Model: OpenCV Zoo YOLOX-S, Apache-2.0; source code in this local package.</footer></main></html>'''
    (out/'report.html').write_text(doc,encoding='utf-8')


def run(args):
    if args.start < 0 or args.end <= args.start or args.threads < 1:
        raise ValueError('Invalid frame range or thread count')
    if args.output.resolve()==args.video.resolve().parent:
        raise ValueError('Output directory must differ from source assets directory')
    verify_annotation_source(args.labels,args.video)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(args.threads)
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise ValueError('Cannot open source video')
    fps=cap.get(cv2.CAP_PROP_FPS)
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or args.end>count:raise ValueError('Range exceeds source video')
    cap.set(cv2.CAP_PROP_POS_FRAMES,args.start)
    ok,first=cap.read()
    if not ok:raise ValueError('Cannot read first frame')
    start=time.perf_counter();detector=Detector(args.model);load=time.perf_counter()-start
    start=time.perf_counter();detector(first);warm=time.perf_counter()-start
    tracker=Tracker()
    h,w=first.shape[:2]
    intermediate=out/'annotated-working.mp4'
    writer=cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h+80))
    if not writer.isOpened():raise ValueError('Cannot open video writer')
    infer_times=[];track_times=[];records=[]
    loop_start=time.perf_counter()
    try:
        for index in range(args.start,args.end):
            if index==args.start:frame=first
            else:
                ok,frame=cap.read()
                if not ok:raise ValueError(f'Unexpected end of source at {index}')
            start=time.perf_counter();detections=detector(frame);infer_times.append(time.perf_counter()-start)
            start=time.perf_counter();tracks=tracker.update(index,detections);track_times.append(time.perf_counter()-start)
            records.append({'source_frame':index,'timestamp_s':index/fps,'detections':detections,'tracks':tracks,'detector_ms':infer_times[-1]*1000,'tracker_ms':track_times[-1]*1000})
            draw=frame.copy()
            for d in tracks:
                a,b,c,e=map(int,d['box']);ident=d['track_id']
                color=((ident*73)%180+65,(ident*131)%180+65,(ident*47)%180+65)
                cv2.rectangle(draw,(a,b),(c,e),color,2)
                cv2.putText(draw,f'T{ident} {d["score"]:.2f}',(max(0,a),max(16,b-5)),0,.45,color,1,cv2.LINE_AA)
            canvas=np.full((h+80,w,3),(35,27,16),dtype=np.uint8);canvas[:h]=draw
            cv2.putText(canvas,'TOUCHLINE  /  person tracking baseline',(15,h+26),0,.55,(177,222,103),1,cv2.LINE_AA)
            cv2.putText(canvas,f'frame {index} | {len(tracks)} observed tracks | CPU detector {infer_times[-1]*1000:.0f} ms',(15,h+49),0,.42,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(canvas,'University of Auckland / CC BY 2.5 | IDs are track labels, not player identities',(15,h+69),0,.34,(180,180,180),1,cv2.LINE_AA)
            writer.write(canvas)
            if index==args.start+min(20,args.end-args.start-1):cv2.imwrite(str(out/'preview.jpg'),canvas)
    finally:
        writer.release();cap.release()
    elapsed=time.perf_counter()-loop_start
    import imageio_ffmpeg
    transcode_start=time.perf_counter()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(intermediate),'-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'annotated.mp4')],check=True,capture_output=True)
    transcode_elapsed=time.perf_counter()-transcode_start
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    stats={'frames':len(records),'start_frame':args.start,'end_frame_exclusive':args.end,'source_fps':fps,'dimensions':[w,h],
           'model_load_ms':load*1000,'warmup_ms':warm*1000,'detector':distribution(infer_times),'tracker':distribution(track_times),
           'loop_elapsed_s':elapsed,'loop_fps':len(records)/elapsed,'h264_transcode_s':transcode_elapsed,'opencv_threads':args.threads,
           'platform':platform.platform(),'processor':platform.processor(),'python':platform.python_version(),'opencv':cv2.__version__,'numpy':np.__version__,
           'source_sha256':digest(args.video),'model_sha256':digest(args.model),
           'settings':{'detector_confidence':0.15,'nms':0.5,'tracker_high':0.5,'tracker_low':0.15,'tracker_iou':0.15,'max_age_source_frames':10},
           'source_files':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}}
    metrics=evaluate_files(args.labels,out/'predictions.jsonl')
    # Detection quality is evaluated on the raw detector output, separately from tracked observations.
    label_doc=json.loads(args.labels.read_text())
    from .evaluate import evaluate
    detection_records=[{'source_frame':r['source_frame'],'tracks':[dict(d,track_id=i) for i,d in enumerate(r['detections'])]} for r in records]
    raw=evaluate(label_doc,detection_records)
    metrics['tracked_detection_precision']=metrics['precision'];metrics['tracked_detection_recall']=metrics['recall']
    metrics['precision'],metrics['recall']=raw['precision'],raw['recall']
    metrics['tp'],metrics['fp'],metrics['fn']=raw['tp'],raw['fp'],raw['fn']
    for row,raw_row in zip(metrics['per_frame'],raw['per_frame']):
        row.update({k:raw_row[k] for k in ['predicted','tp','fp','fn']})
    metrics['note']='Precision/recall/TP/FP/FN use raw detections; ID switches/links use track observations. See frozen protocol.'
    (out/'run.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    make_report(out,stats,metrics)
    print(json.dumps({'output':str(out),'frames':len(records),'precision':metrics['precision'],'recall':metrics['recall'],'id_switches':metrics['id_switches'],'loop_fps':stats['loop_fps']},indent=2))


def main():
    parser=argparse.ArgumentParser(description='Local rugby person detection/tracking baseline')
    parser.add_argument('--video',type=Path,default=ROOT/'assets/rugby.mp4')
    parser.add_argument('--model',type=Path,default=ROOT/'assets/yolox.onnx')
    parser.add_argument('--labels',type=Path,default=ROOT/'evaluation/labels.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--start',type=int,default=550)
    parser.add_argument('--end',type=int,default=610)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    run(args)


if __name__=='__main__':main()
