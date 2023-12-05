mex -v -c Fitline/LFLineFitter.cpp -I.
mex -v -c Fitline/LFLineSegment.cpp -I.
mex -v -c Image/DistanceTransform.cpp -I.
mex -v -c Fdcm/EIEdgeImage.cpp -I.
mex -v -c Fdcm/LMDirectionalIntegralDistanceImage.cpp -I.
mex -v -c Fdcm/LMDisplay.cpp -I.
mex -v -c Fdcm/LMDistanceImage.cpp -I.
mex -v -c Fdcm/LMLineMatcher.cpp -I.
mex -v -c Fdcm/LMNonMaximumSuppression.cpp -I.
mex -v -c Fdcm/MatchingCostMap.cpp -I.

% make fdcm
mex -v mex_fdcm_detect.cpp...
    LFLineFitter.o LFLineSegment.o DistanceTransform.o ...
    EIEdgeImage.o LMDirectionalIntegralDistanceImage.o LMDisplay.o...
    LMDistanceImage.o LMLineMatcher.o LMNonMaximumSuppression.o...
    MatchingCostMap.o -I.

% make fitline
mex -v mex_fitline.cpp LFLineFitter.o LFLineSegment.o -I.

delete *.o