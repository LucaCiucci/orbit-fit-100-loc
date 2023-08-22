use super::*;

const DT: f64 = 0.25;

pub fn fit_trajectory(observations: &Vec<f64>) -> (State<f64>, MinimizationReport<f64>) {
    let guess_position = |angle: f64| Vector2::new(angle.cos(), angle.sin());
    let initial_guess = State {
        pos: guess_position(observations[0]),
        vel: (guess_position(observations[1]) - guess_position(observations[0])) / DT,
    };
    let problem = OptimizationProblem {
        p: initial_guess,
        observed: observations,
    };
    let (result, report) = LevenbergMarquardt::new().minimize(problem);
    (result.p, report)
}

#[derive(Debug, Clone)]
pub struct State<T = f64> {
    pub pos: Vector2<T>,
    pub vel: Vector2<T>,
}

pub fn integrate_trajectory_euler<T>(initial_state: &State<T>) -> impl Iterator<Item = State<T>>
where
    T: Real + Debug + AddAssign + DivAssign + MulAssign + 'static,
{
    let mut state = initial_state.clone();
    let dt = T::from(DT).unwrap();
    std::iter::from_fn(move || {
        let dist2 = state.pos[0].powi(2) + state.pos[1].powi(2);
        let acc = -state.pos / dist2.sqrt().powi(3);
        state.pos += state.vel * dt;
        state.vel += acc * dt;
        Some(state.clone())
    }).take(120)
}

pub fn sampled_trajectory<T>(initial_state: &State<T>) -> impl Iterator<Item = Vector2<T>>
where
    T: Real + Debug + AddAssign + DivAssign + MulAssign + 'static,
{
    integrate_trajectory_euler(initial_state).step_by(5).map(|s| s.pos)
}

pub fn observe<'a, T>(sampled_trajectory: &'a [Vector2<T>]) -> impl Iterator<Item = T> + 'a
where
    T: Real + Debug + AddAssign + DivAssign + MulAssign + 'static,
{
    sampled_trajectory.iter().map(|p| p[1].atan2(p[0]))
}

struct OptimizationProblem<'a> {
    p: State<f64>,
    observed: &'a Vec<f64>,
}

impl<'a> OptimizationProblem<'a> {
    fn residuals<T>(&self, initial_state: &State<T>) -> Vec<T>
    where
        T: Real + Debug + AddAssign + DivAssign + MulAssign + 'static,
    {
        let sampled_trajectory = sampled_trajectory(&initial_state).collect::<Vec<_>>();
        let predicted = observe(&sampled_trajectory).collect::<Vec<_>>();
        self.observed.iter().zip(predicted.iter()).map(|(o, p)| T::from(*o).unwrap() - *p).collect::<Vec<_>>()
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U4> for OptimizationProblem<'a> {
    type ResidualStorage = nalgebra::storage::Owned<f64, Dyn>;
    type JacobianStorage = nalgebra::storage::Owned<f64, Dyn, U4>;
    type ParameterStorage = nalgebra::storage::Owned<f64, U4>;
    fn set_params(&mut self, x: &nalgebra::Vector<f64, U4, Self::ParameterStorage>) {
        self.p.pos[0] = x[0];
        self.p.pos[1] = x[1];
        self.p.vel[0] = x[2];
        self.p.vel[1] = x[3];
    }
    fn params(&self) -> nalgebra::Vector<f64, U4, Self::ParameterStorage> {
        nalgebra::Vector::<f64, U4, Self::ParameterStorage>::new(
            self.p.pos[0],
            self.p.pos[1],
            self.p.vel[0],
            self.p.vel[1],
        )
    }
    fn residuals(&self) -> Option<nalgebra::Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(nalgebra::Vector::<f64, Dyn, Self::ResidualStorage>::from_vec(self.residuals(&self.p)))
    }
    fn jacobian(&self) -> Option<nalgebra::Matrix<f64, Dyn, U4, Self::JacobianStorage>> {
        let mut state = State::<Differential<f64, Vector4<f64>>> {
            pos: Vector2::new(
                self.p.pos[0].into(),
                self.p.pos[1].into(),
            ),
            vel: Vector2::new(
                self.p.vel[0].into(),
                self.p.vel[1].into(),
            ),
        };
        state.pos[0].derivative[0] = 1.0;
        state.pos[1].derivative[1] = 1.0;
        state.vel[0].derivative[2] = 1.0;
        state.vel[1].derivative[3] = 1.0;
        let residuals = self.residuals(&state);
        let mut jacobian = nalgebra::Matrix::<f64, Dyn, U4, Self::JacobianStorage>::zeros_generic(Dyn(residuals.len()), U4::name());
        for (i, r) in residuals.iter().enumerate() {
            jacobian[(i, 0)] = r.derivative[0];
            jacobian[(i, 1)] = r.derivative[1];
            jacobian[(i, 2)] = r.derivative[2];
            jacobian[(i, 3)] = r.derivative[3];
        }
        Some(jacobian)
    }
}