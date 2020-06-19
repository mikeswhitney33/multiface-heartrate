# Multi person no contact heart rate detection

We use face detection + EVM to extend the 2012 work of eulerian video magnification. We essentially take portions of the video with faces in them and perform traditional EVM. This gives a robustness not found in the original work. 

The dependencies are a bit weird, but should be up to date in requirements.txt (no guarantees!)

We wrote a short paper https://andrewnc.github.io/heartrate.pdf and have a companion video https://andrewnc.github.io/projects/projects.html#heart-rate

It runs on a single laptop (no gpu) on a commodity web camera. It is accurate within 6 bpm and is directionally perfect. 

This was a great project. 
