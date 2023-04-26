(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o255 o278 o445 o720 o991)

(:init
    (box-empty)
    (unpacked o255)
    (unpacked o278)
    (unpacked o445)
    (unpacked o720)
    (unpacked o991)
    (heavier o445 o991)
    (heavier o445 o255)
    (heavier o445 o720)
    (heavier o445 o278)
    (heavier o991 o255)
    (heavier o991 o720)
    (heavier o991 o278)
    (heavier o255 o720)
    (heavier o255 o278)
    (heavier o720 o278)
)

(:goal (and (packed o255) (packed o278) (packed o445) (packed o720) (packed o991)))
)
