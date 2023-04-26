(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o107 o224 o318 o471 o477 o569 o694 o701 o908 o925)

(:init
    (box-empty)
    (unpacked o107)
    (unpacked o224)
    (unpacked o318)
    (unpacked o471)
    (unpacked o477)
    (unpacked o569)
    (unpacked o694)
    (unpacked o701)
    (unpacked o908)
    (unpacked o925)
    (heavier o569 o471)
    (heavier o569 o694)
    (heavier o569 o107)
    (heavier o569 o224)
    (heavier o569 o318)
    (heavier o569 o477)
    (heavier o569 o925)
    (heavier o569 o908)
    (heavier o569 o701)
    (heavier o471 o694)
    (heavier o471 o107)
    (heavier o471 o224)
    (heavier o471 o318)
    (heavier o471 o477)
    (heavier o471 o925)
    (heavier o471 o908)
    (heavier o471 o701)
    (heavier o694 o107)
    (heavier o694 o224)
    (heavier o694 o318)
    (heavier o694 o477)
    (heavier o694 o925)
    (heavier o694 o908)
    (heavier o694 o701)
    (heavier o107 o224)
    (heavier o107 o318)
    (heavier o107 o477)
    (heavier o107 o925)
    (heavier o107 o908)
    (heavier o107 o701)
    (heavier o224 o318)
    (heavier o224 o477)
    (heavier o224 o925)
    (heavier o224 o908)
    (heavier o224 o701)
    (heavier o318 o477)
    (heavier o318 o925)
    (heavier o318 o908)
    (heavier o318 o701)
    (heavier o477 o925)
    (heavier o477 o908)
    (heavier o477 o701)
    (heavier o925 o908)
    (heavier o925 o701)
    (heavier o908 o701)
)

(:goal (and (packed o107) (packed o224) (packed o318) (packed o471) (packed o477) (packed o569) (packed o694) (packed o701) (packed o908) (packed o925)))
)
