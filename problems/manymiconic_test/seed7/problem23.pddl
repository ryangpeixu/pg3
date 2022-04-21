(define (problem miconicprob)
	(:domain miconic)
	(:objects
		f0_b0 - floor
		f1_b0 - floor
		f2_b0 - floor
		f3_b0 - floor
		f4_b0 - floor
		f5_b0 - floor
		f6_b0 - floor
		f7_b0 - floor
		f8_b0 - floor
		f9_b0 - floor
		f10_b0 - floor
		p0_b0 - passenger
		p1_b0 - passenger
		p2_b0 - passenger
		p3_b0 - passenger
		p4_b0 - passenger
		p5_b0 - passenger
		p6_b0 - passenger
		f0_b1 - floor
		f1_b1 - floor
		f2_b1 - floor
		f3_b1 - floor
		f4_b1 - floor
		f5_b1 - floor
		f6_b1 - floor
		f7_b1 - floor
		f8_b1 - floor
		f9_b1 - floor
		f10_b1 - floor
		p0_b1 - passenger
		p1_b1 - passenger
		p2_b1 - passenger
		p3_b1 - passenger
		p4_b1 - passenger
		p5_b1 - passenger
		p6_b1 - passenger
		f0_b2 - floor
		f1_b2 - floor
		f2_b2 - floor
		f3_b2 - floor
		f4_b2 - floor
		f5_b2 - floor
		f6_b2 - floor
		f7_b2 - floor
		f8_b2 - floor
		f9_b2 - floor
		f10_b2 - floor
		p0_b2 - passenger
		p1_b2 - passenger
		p2_b2 - passenger
		p3_b2 - passenger
		p4_b2 - passenger
		p5_b2 - passenger
		p6_b2 - passenger
		f0_b3 - floor
		f1_b3 - floor
		f2_b3 - floor
		f3_b3 - floor
		f4_b3 - floor
		f5_b3 - floor
		f6_b3 - floor
		f7_b3 - floor
		f8_b3 - floor
		f9_b3 - floor
		f10_b3 - floor
		p0_b3 - passenger
		p1_b3 - passenger
		p2_b3 - passenger
		p3_b3 - passenger
		p4_b3 - passenger
		p5_b3 - passenger
		p6_b3 - passenger
		f0_b4 - floor
		f1_b4 - floor
		f2_b4 - floor
		f3_b4 - floor
		f4_b4 - floor
		f5_b4 - floor
		f6_b4 - floor
		f7_b4 - floor
		f8_b4 - floor
		f9_b4 - floor
		f10_b4 - floor
		p0_b4 - passenger
		p1_b4 - passenger
		p2_b4 - passenger
		p3_b4 - passenger
		p4_b4 - passenger
		p5_b4 - passenger
		p6_b4 - passenger
	)

(:init
	(above f0_b0 f1_b0)
	(above f0_b0 f2_b0)
	(above f0_b0 f3_b0)
	(above f0_b0 f4_b0)
	(above f0_b0 f5_b0)
	(above f0_b0 f6_b0)
	(above f0_b0 f7_b0)
	(above f0_b0 f8_b0)
	(above f0_b0 f9_b0)
	(above f0_b0 f10_b0)
	(above f1_b0 f2_b0)
	(above f1_b0 f3_b0)
	(above f1_b0 f4_b0)
	(above f1_b0 f5_b0)
	(above f1_b0 f6_b0)
	(above f1_b0 f7_b0)
	(above f1_b0 f8_b0)
	(above f1_b0 f9_b0)
	(above f1_b0 f10_b0)
	(above f2_b0 f3_b0)
	(above f2_b0 f4_b0)
	(above f2_b0 f5_b0)
	(above f2_b0 f6_b0)
	(above f2_b0 f7_b0)
	(above f2_b0 f8_b0)
	(above f2_b0 f9_b0)
	(above f2_b0 f10_b0)
	(above f3_b0 f4_b0)
	(above f3_b0 f5_b0)
	(above f3_b0 f6_b0)
	(above f3_b0 f7_b0)
	(above f3_b0 f8_b0)
	(above f3_b0 f9_b0)
	(above f3_b0 f10_b0)
	(above f4_b0 f5_b0)
	(above f4_b0 f6_b0)
	(above f4_b0 f7_b0)
	(above f4_b0 f8_b0)
	(above f4_b0 f9_b0)
	(above f4_b0 f10_b0)
	(above f5_b0 f6_b0)
	(above f5_b0 f7_b0)
	(above f5_b0 f8_b0)
	(above f5_b0 f9_b0)
	(above f5_b0 f10_b0)
	(above f6_b0 f7_b0)
	(above f6_b0 f8_b0)
	(above f6_b0 f9_b0)
	(above f6_b0 f10_b0)
	(above f7_b0 f8_b0)
	(above f7_b0 f9_b0)
	(above f7_b0 f10_b0)
	(above f8_b0 f9_b0)
	(above f8_b0 f10_b0)
	(above f9_b0 f10_b0)
	(above f0_b1 f1_b1)
	(above f0_b1 f2_b1)
	(above f0_b1 f3_b1)
	(above f0_b1 f4_b1)
	(above f0_b1 f5_b1)
	(above f0_b1 f6_b1)
	(above f0_b1 f7_b1)
	(above f0_b1 f8_b1)
	(above f0_b1 f9_b1)
	(above f0_b1 f10_b1)
	(above f1_b1 f2_b1)
	(above f1_b1 f3_b1)
	(above f1_b1 f4_b1)
	(above f1_b1 f5_b1)
	(above f1_b1 f6_b1)
	(above f1_b1 f7_b1)
	(above f1_b1 f8_b1)
	(above f1_b1 f9_b1)
	(above f1_b1 f10_b1)
	(above f2_b1 f3_b1)
	(above f2_b1 f4_b1)
	(above f2_b1 f5_b1)
	(above f2_b1 f6_b1)
	(above f2_b1 f7_b1)
	(above f2_b1 f8_b1)
	(above f2_b1 f9_b1)
	(above f2_b1 f10_b1)
	(above f3_b1 f4_b1)
	(above f3_b1 f5_b1)
	(above f3_b1 f6_b1)
	(above f3_b1 f7_b1)
	(above f3_b1 f8_b1)
	(above f3_b1 f9_b1)
	(above f3_b1 f10_b1)
	(above f4_b1 f5_b1)
	(above f4_b1 f6_b1)
	(above f4_b1 f7_b1)
	(above f4_b1 f8_b1)
	(above f4_b1 f9_b1)
	(above f4_b1 f10_b1)
	(above f5_b1 f6_b1)
	(above f5_b1 f7_b1)
	(above f5_b1 f8_b1)
	(above f5_b1 f9_b1)
	(above f5_b1 f10_b1)
	(above f6_b1 f7_b1)
	(above f6_b1 f8_b1)
	(above f6_b1 f9_b1)
	(above f6_b1 f10_b1)
	(above f7_b1 f8_b1)
	(above f7_b1 f9_b1)
	(above f7_b1 f10_b1)
	(above f8_b1 f9_b1)
	(above f8_b1 f10_b1)
	(above f9_b1 f10_b1)
	(above f0_b2 f1_b2)
	(above f0_b2 f2_b2)
	(above f0_b2 f3_b2)
	(above f0_b2 f4_b2)
	(above f0_b2 f5_b2)
	(above f0_b2 f6_b2)
	(above f0_b2 f7_b2)
	(above f0_b2 f8_b2)
	(above f0_b2 f9_b2)
	(above f0_b2 f10_b2)
	(above f1_b2 f2_b2)
	(above f1_b2 f3_b2)
	(above f1_b2 f4_b2)
	(above f1_b2 f5_b2)
	(above f1_b2 f6_b2)
	(above f1_b2 f7_b2)
	(above f1_b2 f8_b2)
	(above f1_b2 f9_b2)
	(above f1_b2 f10_b2)
	(above f2_b2 f3_b2)
	(above f2_b2 f4_b2)
	(above f2_b2 f5_b2)
	(above f2_b2 f6_b2)
	(above f2_b2 f7_b2)
	(above f2_b2 f8_b2)
	(above f2_b2 f9_b2)
	(above f2_b2 f10_b2)
	(above f3_b2 f4_b2)
	(above f3_b2 f5_b2)
	(above f3_b2 f6_b2)
	(above f3_b2 f7_b2)
	(above f3_b2 f8_b2)
	(above f3_b2 f9_b2)
	(above f3_b2 f10_b2)
	(above f4_b2 f5_b2)
	(above f4_b2 f6_b2)
	(above f4_b2 f7_b2)
	(above f4_b2 f8_b2)
	(above f4_b2 f9_b2)
	(above f4_b2 f10_b2)
	(above f5_b2 f6_b2)
	(above f5_b2 f7_b2)
	(above f5_b2 f8_b2)
	(above f5_b2 f9_b2)
	(above f5_b2 f10_b2)
	(above f6_b2 f7_b2)
	(above f6_b2 f8_b2)
	(above f6_b2 f9_b2)
	(above f6_b2 f10_b2)
	(above f7_b2 f8_b2)
	(above f7_b2 f9_b2)
	(above f7_b2 f10_b2)
	(above f8_b2 f9_b2)
	(above f8_b2 f10_b2)
	(above f9_b2 f10_b2)
	(above f0_b3 f1_b3)
	(above f0_b3 f2_b3)
	(above f0_b3 f3_b3)
	(above f0_b3 f4_b3)
	(above f0_b3 f5_b3)
	(above f0_b3 f6_b3)
	(above f0_b3 f7_b3)
	(above f0_b3 f8_b3)
	(above f0_b3 f9_b3)
	(above f0_b3 f10_b3)
	(above f1_b3 f2_b3)
	(above f1_b3 f3_b3)
	(above f1_b3 f4_b3)
	(above f1_b3 f5_b3)
	(above f1_b3 f6_b3)
	(above f1_b3 f7_b3)
	(above f1_b3 f8_b3)
	(above f1_b3 f9_b3)
	(above f1_b3 f10_b3)
	(above f2_b3 f3_b3)
	(above f2_b3 f4_b3)
	(above f2_b3 f5_b3)
	(above f2_b3 f6_b3)
	(above f2_b3 f7_b3)
	(above f2_b3 f8_b3)
	(above f2_b3 f9_b3)
	(above f2_b3 f10_b3)
	(above f3_b3 f4_b3)
	(above f3_b3 f5_b3)
	(above f3_b3 f6_b3)
	(above f3_b3 f7_b3)
	(above f3_b3 f8_b3)
	(above f3_b3 f9_b3)
	(above f3_b3 f10_b3)
	(above f4_b3 f5_b3)
	(above f4_b3 f6_b3)
	(above f4_b3 f7_b3)
	(above f4_b3 f8_b3)
	(above f4_b3 f9_b3)
	(above f4_b3 f10_b3)
	(above f5_b3 f6_b3)
	(above f5_b3 f7_b3)
	(above f5_b3 f8_b3)
	(above f5_b3 f9_b3)
	(above f5_b3 f10_b3)
	(above f6_b3 f7_b3)
	(above f6_b3 f8_b3)
	(above f6_b3 f9_b3)
	(above f6_b3 f10_b3)
	(above f7_b3 f8_b3)
	(above f7_b3 f9_b3)
	(above f7_b3 f10_b3)
	(above f8_b3 f9_b3)
	(above f8_b3 f10_b3)
	(above f9_b3 f10_b3)
	(above f0_b4 f1_b4)
	(above f0_b4 f2_b4)
	(above f0_b4 f3_b4)
	(above f0_b4 f4_b4)
	(above f0_b4 f5_b4)
	(above f0_b4 f6_b4)
	(above f0_b4 f7_b4)
	(above f0_b4 f8_b4)
	(above f0_b4 f9_b4)
	(above f0_b4 f10_b4)
	(above f1_b4 f2_b4)
	(above f1_b4 f3_b4)
	(above f1_b4 f4_b4)
	(above f1_b4 f5_b4)
	(above f1_b4 f6_b4)
	(above f1_b4 f7_b4)
	(above f1_b4 f8_b4)
	(above f1_b4 f9_b4)
	(above f1_b4 f10_b4)
	(above f2_b4 f3_b4)
	(above f2_b4 f4_b4)
	(above f2_b4 f5_b4)
	(above f2_b4 f6_b4)
	(above f2_b4 f7_b4)
	(above f2_b4 f8_b4)
	(above f2_b4 f9_b4)
	(above f2_b4 f10_b4)
	(above f3_b4 f4_b4)
	(above f3_b4 f5_b4)
	(above f3_b4 f6_b4)
	(above f3_b4 f7_b4)
	(above f3_b4 f8_b4)
	(above f3_b4 f9_b4)
	(above f3_b4 f10_b4)
	(above f4_b4 f5_b4)
	(above f4_b4 f6_b4)
	(above f4_b4 f7_b4)
	(above f4_b4 f8_b4)
	(above f4_b4 f9_b4)
	(above f4_b4 f10_b4)
	(above f5_b4 f6_b4)
	(above f5_b4 f7_b4)
	(above f5_b4 f8_b4)
	(above f5_b4 f9_b4)
	(above f5_b4 f10_b4)
	(above f6_b4 f7_b4)
	(above f6_b4 f8_b4)
	(above f6_b4 f9_b4)
	(above f6_b4 f10_b4)
	(above f7_b4 f8_b4)
	(above f7_b4 f9_b4)
	(above f7_b4 f10_b4)
	(above f8_b4 f9_b4)
	(above f8_b4 f10_b4)
	(above f9_b4 f10_b4)

	(origin p0_b0 f7_b0)
	(destin p0_b0 f8_b0)
	(origin p1_b0 f3_b0)
	(destin p1_b0 f1_b0)
	(origin p2_b0 f4_b0)
	(destin p2_b0 f2_b0)
	(origin p3_b0 f2_b0)
	(destin p3_b0 f9_b0)
	(origin p4_b0 f9_b0)
	(destin p4_b0 f1_b0)
	(origin p5_b0 f10_b0)
	(destin p5_b0 f7_b0)
	(origin p6_b0 f0_b0)
	(destin p6_b0 f9_b0)
	(origin p0_b1 f0_b1)
	(destin p0_b1 f1_b1)
	(origin p1_b1 f3_b1)
	(destin p1_b1 f4_b1)
	(origin p2_b1 f1_b1)
	(destin p2_b1 f5_b1)
	(origin p3_b1 f5_b1)
	(destin p3_b1 f4_b1)
	(origin p4_b1 f8_b1)
	(destin p4_b1 f6_b1)
	(origin p5_b1 f10_b1)
	(destin p5_b1 f1_b1)
	(origin p6_b1 f2_b1)
	(destin p6_b1 f1_b1)
	(origin p0_b2 f4_b2)
	(destin p0_b2 f9_b2)
	(origin p1_b2 f10_b2)
	(destin p1_b2 f3_b2)
	(origin p2_b2 f7_b2)
	(destin p2_b2 f4_b2)
	(origin p3_b2 f3_b2)
	(destin p3_b2 f10_b2)
	(origin p4_b2 f0_b2)
	(destin p4_b2 f1_b2)
	(origin p5_b2 f8_b2)
	(destin p5_b2 f4_b2)
	(origin p6_b2 f6_b2)
	(destin p6_b2 f4_b2)
	(origin p0_b3 f5_b3)
	(destin p0_b3 f3_b3)
	(origin p1_b3 f8_b3)
	(destin p1_b3 f9_b3)
	(origin p2_b3 f3_b3)
	(destin p2_b3 f9_b3)
	(origin p3_b3 f4_b3)
	(destin p3_b3 f0_b3)
	(origin p4_b3 f6_b3)
	(destin p4_b3 f4_b3)
	(origin p5_b3 f0_b3)
	(destin p5_b3 f1_b3)
	(origin p6_b3 f2_b3)
	(destin p6_b3 f8_b3)
	(origin p0_b4 f10_b4)
	(destin p0_b4 f6_b4)
	(origin p1_b4 f1_b4)
	(destin p1_b4 f5_b4)
	(origin p2_b4 f4_b4)
	(destin p2_b4 f7_b4)
	(origin p3_b4 f7_b4)
	(destin p3_b4 f3_b4)
	(origin p4_b4 f5_b4)
	(destin p4_b4 f0_b4)
	(origin p5_b4 f9_b4)
	(destin p5_b4 f5_b4)
	(origin p6_b4 f6_b4)
	(destin p6_b4 f5_b4)

	(lift-at f10_b0)
	(lift-at f6_b1)
	(lift-at f3_b2)
	(lift-at f0_b3)
	(lift-at f4_b4)
)

(:goal (and
	(served p0_b0)
	(served p1_b0)
	(served p2_b0)
	(served p3_b0)
	(served p4_b0)
	(served p5_b0)
	(served p6_b0)
	(served p0_b1)
	(served p1_b1)
	(served p2_b1)
	(served p3_b1)
	(served p4_b1)
	(served p5_b1)
	(served p6_b1)
	(served p0_b2)
	(served p1_b2)
	(served p2_b2)
	(served p3_b2)
	(served p4_b2)
	(served p5_b2)
	(served p6_b2)
	(served p0_b3)
	(served p1_b3)
	(served p2_b3)
	(served p3_b3)
	(served p4_b3)
	(served p5_b3)
	(served p6_b3)
	(served p0_b4)
	(served p1_b4)
	(served p2_b4)
	(served p3_b4)
	(served p4_b4)
	(served p5_b4)
	(served p6_b4)
))