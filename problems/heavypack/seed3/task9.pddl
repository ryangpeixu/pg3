(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o205 o251 o476 o589 o851 o922 o941)

(:init
    (box-empty)
    (unpacked o205)
    (unpacked o251)
    (unpacked o476)
    (unpacked o589)
    (unpacked o851)
    (unpacked o922)
    (unpacked o941)
    (heavier o941 o476)
    (heavier o941 o922)
    (heavier o941 o251)
    (heavier o941 o589)
    (heavier o941 o205)
    (heavier o941 o851)
    (heavier o476 o922)
    (heavier o476 o251)
    (heavier o476 o589)
    (heavier o476 o205)
    (heavier o476 o851)
    (heavier o922 o251)
    (heavier o922 o589)
    (heavier o922 o205)
    (heavier o922 o851)
    (heavier o251 o589)
    (heavier o251 o205)
    (heavier o251 o851)
    (heavier o589 o205)
    (heavier o589 o851)
    (heavier o205 o851)
)

(:goal (and (packed o205) (packed o251) (packed o476) (packed o589) (packed o851) (packed o922) (packed o941)))
)