(define (problem heavy-pack-prob)
	(:domain heavy-pack)
	(:objects o150 o239 o358 o440 o547 o562 o677 o957)

(:init
    (box-empty)
    (unpacked o150)
    (unpacked o239)
    (unpacked o358)
    (unpacked o440)
    (unpacked o547)
    (unpacked o562)
    (unpacked o677)
    (unpacked o957)
    (heavier o677 o547)
    (heavier o677 o358)
    (heavier o677 o562)
    (heavier o677 o150)
    (heavier o677 o239)
    (heavier o677 o440)
    (heavier o677 o957)
    (heavier o547 o358)
    (heavier o547 o562)
    (heavier o547 o150)
    (heavier o547 o239)
    (heavier o547 o440)
    (heavier o547 o957)
    (heavier o358 o562)
    (heavier o358 o150)
    (heavier o358 o239)
    (heavier o358 o440)
    (heavier o358 o957)
    (heavier o562 o150)
    (heavier o562 o239)
    (heavier o562 o440)
    (heavier o562 o957)
    (heavier o150 o239)
    (heavier o150 o440)
    (heavier o150 o957)
    (heavier o239 o440)
    (heavier o239 o957)
    (heavier o440 o957)
)

(:goal (and (packed o150) (packed o239) (packed o358) (packed o440) (packed o547) (packed o562) (packed o677) (packed o957)))
)