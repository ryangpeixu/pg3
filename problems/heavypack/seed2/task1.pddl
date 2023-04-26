(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o150 o260 o433 o669 o679 o749 o946)

(:init
    (box-empty)
    (unpacked o150)
    (unpacked o260)
    (unpacked o433)
    (unpacked o669)
    (unpacked o679)
    (unpacked o749)
    (unpacked o946)
    (heavier o260 o669)
    (heavier o260 o433)
    (heavier o260 o679)
    (heavier o260 o946)
    (heavier o260 o749)
    (heavier o260 o150)
    (heavier o669 o433)
    (heavier o669 o679)
    (heavier o669 o946)
    (heavier o669 o749)
    (heavier o669 o150)
    (heavier o433 o679)
    (heavier o433 o946)
    (heavier o433 o749)
    (heavier o433 o150)
    (heavier o679 o946)
    (heavier o679 o749)
    (heavier o679 o150)
    (heavier o946 o749)
    (heavier o946 o150)
    (heavier o749 o150)
)

(:goal (and (packed o150) (packed o260) (packed o433) (packed o669) (packed o679) (packed o749) (packed o946)))
)