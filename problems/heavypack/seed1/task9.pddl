(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o215 o470 o639 o693 o845)

(:init
    (box-empty)
    (unpacked o215)
    (unpacked o470)
    (unpacked o639)
    (unpacked o693)
    (unpacked o845)
    (heavier o215 o470)
    (heavier o215 o639)
    (heavier o215 o845)
    (heavier o215 o693)
    (heavier o470 o639)
    (heavier o470 o845)
    (heavier o470 o693)
    (heavier o639 o845)
    (heavier o639 o693)
    (heavier o845 o693)
)

(:goal (and (packed o215) (packed o470) (packed o639) (packed o693) (packed o845)))
)