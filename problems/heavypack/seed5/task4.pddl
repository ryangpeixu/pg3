(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o544 o873 o884 o896)

(:init
    (box-empty)
    (unpacked o544)
    (unpacked o873)
    (unpacked o884)
    (unpacked o896)
    (heavier o873 o896)
    (heavier o873 o544)
    (heavier o873 o884)
    (heavier o896 o544)
    (heavier o896 o884)
    (heavier o544 o884)
)

(:goal (and (packed o544) (packed o873) (packed o884) (packed o896)))
)