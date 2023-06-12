mod digital_ret;
mod ret_single;

use pyo3::prelude::*;

#[pymodule]
fn equity_calc(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ret_single::calc_ret_single, m)?)?;
    m.add_function(wrap_pyfunction!(digital_ret::calc_digital_ret, m)?)?;
    Ok(())
}
