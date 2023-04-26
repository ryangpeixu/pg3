(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o109 o261 o298 o335 o414 o451 o600 o815 o92)

(:init
    (box-empty)
    (unpacked o109)
    (unpacked o261)
    (unpacked o298)
    (unpacked o335)
    (unpacked o414)
    (unpacked o451)
    (unpacked o600)
    (unpacked o815)
    (unpacked o92)
    (heavier o451 o815)
    (heavier o451 o600)
    (heavier o451 o92)
    (heavier o451 o298)
    (heavier o451 o109)
    (heavier o451 o414)
    (heavier o451 o261)
    (heavier o451 o335)
    (heavier o815 o600)
    (heavier o815 o92)
    (heavier o815 o298)
    (heavier o815 o109)
    (heavier o815 o414)
    (heavier o815 o261)
    (heavier o815 o335)
    (heavier o600 o92)
    (heavier o600 o298)
    (heavier o600 o109)
    (heavier o600 o414)
    (heavier o600 o261)
    (heavier o600 o335)
    (heavier o92 o298)
    (heavier o92 o109)
    (heavier o92 o414)
    (heavier o92 o261)
    (heavier o92 o335)
    (heavier o298 o109)
    (heavier o298 o414)
    (heavier o298 o261)
    (heavier o298 o335)
    (heavier o109 o414)
    (heavier o109 o261)
    (heavier o109 o335)
    (heavier o414 o261)
    (heavier o414 o335)
    (heavier o261 o335)
)

(:goal (and (packed o109) (packed o261) (packed o298) (packed o335) (packed o414) (packed o451) (packed o600) (packed o815) (packed o92)))
)