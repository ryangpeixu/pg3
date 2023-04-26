(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o123 o330 o452 o51 o618 o672 o680 o756)

(:init
    (box-empty)
    (unpacked o123)
    (unpacked o330)
    (unpacked o452)
    (unpacked o51)
    (unpacked o618)
    (unpacked o672)
    (unpacked o680)
    (unpacked o756)
    (heavier o452 o51)
    (heavier o452 o123)
    (heavier o452 o680)
    (heavier o452 o756)
    (heavier o452 o330)
    (heavier o452 o672)
    (heavier o452 o618)
    (heavier o51 o123)
    (heavier o51 o680)
    (heavier o51 o756)
    (heavier o51 o330)
    (heavier o51 o672)
    (heavier o51 o618)
    (heavier o123 o680)
    (heavier o123 o756)
    (heavier o123 o330)
    (heavier o123 o672)
    (heavier o123 o618)
    (heavier o680 o756)
    (heavier o680 o330)
    (heavier o680 o672)
    (heavier o680 o618)
    (heavier o756 o330)
    (heavier o756 o672)
    (heavier o756 o618)
    (heavier o330 o672)
    (heavier o330 o618)
    (heavier o672 o618)
)

(:goal (and (packed o123) (packed o330) (packed o452) (packed o51) (packed o618) (packed o672) (packed o680) (packed o756)))
)
