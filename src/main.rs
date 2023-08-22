use std::fmt::Debug;
use std::ops::{DivAssign, MulAssign, AddAssign};

use differential::Differential;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, MinimizationReport};
use nalgebra::{Vector2, U4, Dyn, Vector4, DimName};
use num_traits::real::Real;

use plotters::prelude::*;

mod fit; use fit::*;


fn main() {
    println!("Hello, world!");

    let root = SVGBackend::new("out.svg", (500, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();


    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(
            "2D orbit fit example",
            ("sans-serif", 20),
        )
        //.set_label_area_size(LabelAreaPosition::Left, 60)
        //.set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(
            -2.5..7.0,
            -10.0..10.0,
        ).unwrap();

    chart
        .configure_mesh()
        .max_light_lines(4)
        //.y_desc("Average Temp (F)")
        .draw().unwrap();

    chart.draw_series(
        (0..1).map(|_| Circle::new((0.0, 0.0), 10, BLUE.filled())),
    ).unwrap();

    let initial_state = State {
        pos: Vector2::new(3.0, -8.0),
        vel: Vector2::new(0.25, 0.5),
    };
    println!("initial state: {:?}", initial_state);

    let points = integrate_trajectory_euler(&initial_state)
        .map(|s| (s.pos[0], s.pos[1]));

    chart.draw_series(LineSeries::new(
        points,
        &RED,
    )).unwrap()
    .label("actual trajectory")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    fn random_vector() -> Vector2<f64> {
        Vector2::new(rand::random::<f64>() - 0.5, rand::random::<f64>() - 0.5) * 0.5
    }

    let sampled = sampled_trajectory(&initial_state)
        .map(|p| p + random_vector())
        .collect::<Vec<_>>();
    let observed = observe(&sampled).collect::<Vec<_>>();
    chart.draw_series(
        sampled.iter().map(|p| Cross::new((p[0], p[1]), 3, BLACK)),
    ).unwrap()
    .label("observations")
    .legend(|(x, y)| Cross::new((x + 10, y), 5, &BLACK));

    let (computed, report) = fit_trajectory(&observed);
    println!("report: {:?}", report);
    println!("computed state: {:?}", computed);
    let points = integrate_trajectory_euler(&computed)
        .map(|s| (s.pos[0], s.pos[1]));
    chart.draw_series(LineSeries::new(
        points,
        &BLUE,
    )).unwrap()
    .label("computed trajectory")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw().unwrap();

    // benchmark
    /*let start_time = std::time::Instant::now();
    const N: usize = 10000;
    for _ in 0..N {
        let _ = fit_trajectory(&observed);
    }
    println!("fit took {:?}", start_time.elapsed() / N as u32);
    */
}

