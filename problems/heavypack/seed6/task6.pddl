(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o150 o182 o252 o274 o380 o462 o490 o842 o91 o940)

(:init
    (box-empty)
    (unpacked o150)
    (unpacked o182)
    (unpacked o252)
    (unpacked o274)
    (unpacked o380)
    (unpacked o462)
    (unpacked o490)
    (unpacked o842)
    (unpacked o91)
    (unpacked o940)
    (heavier o150 o940)
    (heavier o150 o380)
    (heavier o150 o842)
    (heavier o150 o91)
    (heavier o150 o274)
    (heavier o150 o182)
    (heavier o150 o462)
    (heavier o150 o252)
    (heavier o150 o490)
    (heavier o940 o380)
    (heavier o940 o842)
    (heavier o940 o91)
    (heavier o940 o274)
    (heavier o940 o182)
    (heavier o940 o462)
    (heavier o940 o252)
    (heavier o940 o490)
    (heavier o380 o842)
    (heavier o380 o91)
    (heavier o380 o274)
    (heavier o380 o182)
    (heavier o380 o462)
    (heavier o380 o252)
    (heavier o380 o490)
    (heavier o842 o91)
    (heavier o842 o274)
    (heavier o842 o182)
    (heavier o842 o462)
    (heavier o842 o252)
    (heavier o842 o490)
    (heavier o91 o274)
    (heavier o91 o182)
    (heavier o91 o462)
    (heavier o91 o252)
    (heavier o91 o490)
    (heavier o274 o182)
    (heavier o274 o462)
    (heavier o274 o252)
    (heavier o274 o490)
    (heavier o182 o462)
    (heavier o182 o252)
    (heavier o182 o490)
    (heavier o462 o252)
    (heavier o462 o490)
    (heavier o252 o490)
)

(:goal (and (packed o150) (packed o182) (packed o252) (packed o274) (packed o380) (packed o462) (packed o490) (packed o842) (packed o91) (packed o940)))
)