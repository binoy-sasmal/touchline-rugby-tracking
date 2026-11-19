import numpy as np
from rugby_track.detector import Detector


def test_yolox_geometry_and_rgb_zero_to_255_preprocessing():
    class FakeNet:
        def setInput(self,blob):self.blob=blob
        def forward(self):
            raw=np.zeros((1,8400,85),dtype=np.float32)
            # First anchor has stride 8: centre (80,80), size (40,80).
            raw[0,0,:4]=[10,10,np.log(5),np.log(10)]
            raw[0,0,4]=.9;raw[0,0,5]=.8
            return raw
    detector=Detector.__new__(Detector);detector.net=FakeNet();detector.size=640
    detector.confidence=.15;detector.nms=.5
    detector.grids=np.zeros((8400,2));detector.strides=np.full((8400,1),8)
    frame=np.zeros((320,640,3),dtype=np.uint8);frame[:]=[10,20,30]
    result=detector(frame)
    np.testing.assert_allclose(result[0]['box'],[60,40,100,120],atol=.001)
    np.testing.assert_allclose(detector.net.blob[0,:,0,0],[30,20,10])
    np.testing.assert_allclose(detector.net.blob[0,:,500,0],[114,114,114])
    assert abs(result[0]['score']-.72)<.0001
