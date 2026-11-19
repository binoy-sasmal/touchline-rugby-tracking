"""YOLOX-S ONNX inference. Pre/postprocess follows OpenCV Zoo's YOLOX format.

See assets/LICENSE-YOLOX.txt and THIRD_PARTY.md. No model training or export here.
"""
import cv2
import numpy as np


class Detector:
    def __init__(self, model, confidence=0.15, nms=0.5):
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.confidence, self.nms = confidence, nms
        self.size = 640
        grids, strides = [], []
        for stride in (8,16,32):
            y,x=np.mgrid[:self.size//stride,:self.size//stride]
            grids.append(np.stack([x,y],axis=-1).reshape(-1,2))
            strides.append(np.full((x.size,1),stride))
        self.grids, self.strides = np.concatenate(grids),np.concatenate(strides)

    def __call__(self, frame):
        h,w = frame.shape[:2]
        ratio = min(self.size/h,self.size/w)
        resized=cv2.resize(frame,(int(w*ratio),int(h*ratio)))
        padded=np.full((self.size,self.size,3),114,dtype=np.uint8)
        padded[:resized.shape[0],:resized.shape[1]]=resized
        # Zoo's export takes RGB in 0..255; no /255 normalization.
        blob=cv2.dnn.blobFromImage(padded,1.0,(640,640),swapRB=True,crop=False)
        self.net.setInput(blob)
        raw=self.net.forward()[0].copy()
        if raw.shape != (8400,85):
            raise ValueError(f'Unexpected YOLOX output shape: {raw.shape}')
        raw[:,:2]=(raw[:,:2]+self.grids)*self.strides
        raw[:,2:4]=np.exp(raw[:,2:4])*self.strides
        classes=raw[:,5:].argmax(axis=1)
        scores=raw[:,4]*raw[:,5]  # COCO class zero is person.
        mask=(classes==0)&(scores>=self.confidence)
        raw,scores=raw[mask],scores[mask]
        xywh=np.column_stack([raw[:,:2]-raw[:,2:4]/2,raw[:,2:4]])/ratio
        indices=cv2.dnn.NMSBoxes(xywh.tolist(),scores.tolist(),self.confidence,self.nms)
        detections=[]
        for i in np.asarray(indices).reshape(-1):
            x,y,bw,bh=xywh[i]
            box=[float(np.clip(x,0,w)),float(np.clip(y,0,h)),float(np.clip(x+bw,0,w)),float(np.clip(y+bh,0,h))]
            if box[2]>box[0] and box[3]>box[1]:
                detections.append({'box':box,'score':float(scores[i])})
        return detections
