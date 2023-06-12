use ndarray::Zip;
use pyo3::{pyfunction, FromPyObject, PyAny, PyResult};
use teapy::arr::{Arr1, Expr, Number, WrapNdarray};
use teapy::pylazy::{parse_expr_nocopy, PyExpr};

pub enum CommisionType {
    Percent,
    Absolute,
}

impl<'source> FromPyObject<'source> for CommisionType {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let s: Option<&str> = ob.extract()?;
        let s = s.unwrap_or("percent").to_lowercase();
        let out = match s.as_str() {
            "percent" => CommisionType::Percent,
            "absolute" => CommisionType::Absolute,
            _ => panic!("不支持的手续费类型: {s}, commision_type必须是'percent'或'absolute'"),
        };
        Ok(out)
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature=(pos, opening_cost, closing_cost, init_cash=1_000_000, multiplier=1, leverage=1., slippage=0., ticksize=0., c_rate=3e-4, blowup=false, commision_type=CommisionType::Percent, contract_change_signal=None))]
pub unsafe fn calc_ret_single(
    pos: &PyAny,
    opening_cost: &PyAny,
    closing_cost: &PyAny,
    init_cash: i64,
    multiplier: i32,
    leverage: f64,
    slippage: f64,
    ticksize: f64,
    c_rate: f64,
    blowup: bool,
    commision_type: CommisionType,
    contract_change_signal: Option<&PyAny>,
) -> PyResult<PyExpr> {
    let pos = parse_expr_nocopy(pos)?;
    let opening_cost = parse_expr_nocopy(opening_cost)?;
    let closing_cost = parse_expr_nocopy(closing_cost)?;
    let (obj, obj1, obj2) = (pos.obj(), opening_cost.obj(), closing_cost.obj());
    let (opening_cost, closing_cost) = (opening_cost.cast_f64()?, closing_cost.cast_f64()?);
    let (contract_signal, obj3) = if let Some(contract_signal_obj) = contract_change_signal {
        let contract_signal = parse_expr_nocopy(contract_signal_obj)?;
        let obj = contract_signal.obj();
        (Some(contract_signal.cast_bool()?), obj)
    } else {
        (None, None)
    };
    let out_expr: Expr<f64> = pos.cast_f64()?.chain_view_f(move |pos_arr| {
        let pos_arr = pos_arr.to_dim1().unwrap(); // 当期仓位的1d array
        let opening_cost_expr = opening_cost.eval(); // 开仓成本的1d array
        let closing_cost_expr = closing_cost.eval(); // 平仓价格的1d array
        let opening_cost_arr = opening_cost_expr.view_arr().to_dim1().unwrap();
        let closing_cost_arr = closing_cost_expr.view_arr().to_dim1().unwrap();
        if pos_arr.is_empty() {
            return Arr1::from_vec(vec![]).to_dimd().unwrap().into();
        }
        // 账户变动信息
        let mut cash = init_cash.f64();
        let mut last_pos = pos_arr[0];
        let mut last_lot_num = 0.;
        let mut last_close = closing_cost_arr[0];
        if let Some(contract_signal) = contract_signal {
            let contract_signal_expr = contract_signal.eval();
            let contract_signal_arr = contract_signal_expr.view_arr().to_dim1().unwrap();
            Zip::from(&pos_arr.0)
                .and(&opening_cost_arr.0)
                .and(&closing_cost_arr.0)
                .and(&contract_signal_arr.0)
                .map_collect(|&pos, &opening_cost, &closing_cost, &contract_signal| {
                    if blowup && cash <= 0. {
                        return 0.;
                    }
                    if (last_lot_num != 0.) && (!contract_signal) {
                        // 换月的时候不计算跳开的损益
                        cash += last_lot_num
                            * (opening_cost - last_close)
                            * multiplier.f64()
                            * last_pos.signum();
                    }
                    // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                    if pos != last_pos {
                        // 仓位出现变化
                        // 计算新的理论持仓手数
                        let lot_num = ((cash * leverage * pos.abs())
                            / (multiplier.f64() * opening_cost))
                            .floor();
                        let lot_num_change = if !contract_signal {
                            (lot_num * pos.signum() - last_lot_num * last_pos.signum()).abs()
                        } else {
                            last_lot_num.abs() * 2.
                        };
                        // 扣除手续费变动
                        if let CommisionType::Percent = commision_type {
                            cash -= lot_num_change
                                * multiplier.f64()
                                * (opening_cost * c_rate + slippage * ticksize);
                        } else {
                            cash -=
                                lot_num_change * (c_rate + multiplier.f64() * slippage * ticksize);
                        };
                        // 更新上期持仓手数和持仓头寸
                        last_lot_num = lot_num;
                        last_pos = pos;
                    }
                    // 计算当期损益
                    if last_lot_num != 0. {
                        cash += last_lot_num
                            * (closing_cost - opening_cost)
                            * multiplier.f64()
                            * last_pos.signum();
                    }
                    last_close = closing_cost; // 更新上期收盘价

                    cash
                    // cash
                })
                .wrap()
                .to_dimd()
                .unwrap()
                .into()
        } else {
            // 不考虑合约换月信号的情况
            Zip::from(&pos_arr.0)
                .and(&opening_cost_arr.0)
                .and(&closing_cost_arr.0)
                .map_collect(|&pos, &opening_cost, &closing_cost| {
                    if blowup && cash <= 0. {
                        return 0.;
                    }
                    if last_lot_num != 0. {
                        cash += last_lot_num
                            * (opening_cost - last_close)
                            * multiplier.f64()
                            * last_pos.signum();
                    }
                    // 因为采用pos来判断加减仓，所以杠杆leverage必须是个常量，不应修改leverage的类型
                    if pos != last_pos {
                        // 仓位出现变化
                        // 计算新的理论持仓手数
                        let lot_num = ((cash * leverage * pos.abs())
                            / (multiplier.f64() * opening_cost))
                            .floor();
                        // 扣除手续费变动
                        if let CommisionType::Percent = commision_type {
                            cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum())
                                .abs()
                                * multiplier.f64()
                                * (opening_cost * c_rate + slippage * ticksize);
                        } else {
                            cash -= (lot_num * pos.signum() - last_lot_num * last_pos.signum())
                                .abs()
                                * (c_rate + multiplier.f64() * slippage * ticksize);
                        };
                        // 更新上期持仓手数和持仓头寸
                        last_lot_num = lot_num;
                        last_pos = pos;
                    }
                    // 计算当期损益
                    if last_lot_num != 0. {
                        cash += last_lot_num
                            * (closing_cost - opening_cost)
                            * multiplier.f64()
                            * last_pos.signum();
                    }
                    last_close = closing_cost; // 更新上期收盘价

                    cash
                    // cash
                })
                .wrap()
                .to_dimd()
                .unwrap()
                .into()
        }
    });
    Ok(out_expr
        .to_py(obj)
        .add_obj(obj1)
        .add_obj(obj2)
        .add_obj(obj3))
}
