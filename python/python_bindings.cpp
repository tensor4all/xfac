
#include <carma>
#include <armadillo>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/cast.h>

#include "xfac/tensor/tensor_ci.h"
#include "xfac/grid.h"
#include "xfac/tensor/tensor_ci_2.h"
#include "xfac/tensor/tensor_ci_converter.h"

using namespace std;
using namespace xfac;

namespace py = pybind11;
using namespace pybind11::literals;

using cmpx=complex<double>;

template<typename T>
void declare_TensorCI(py::module &m, std::string typestr) {
    using TensorTraind=TensorTrain<T>;
    py::class_<TensorTraind>(m, ("TensorTrain"s+typestr).c_str())
            .def(py::init<size_t>(), "len"_a)
            .def_readonly("core",  &TensorTraind::M)
            .def("setCoreAt",[](TensorTraind &tt, int i, py::array const& Mi){ tt.M.at(i)=carma::arr_to_cube<T>(Mi); }, "i"_a, "Mi"_a)
            .def("eval", &TensorTraind::eval)
            .def("sum", &TensorTraind::sum)
            .def("sum1", &TensorTraind::sum1)
            .def("overlap", &TensorTraind::overlap, "tt"_a)
            .def("norm2", &TensorTraind::norm2)
            .def("compressSVD",&TensorTraind::compressSVD, "reltol"_a=1e-12, "maxBondDim"_a=0)
            .def("compressLU",&TensorTraind::compressLU, "reltol"_a=1e-12, "maxBondDim"_a=0)
            .def("compressCI",&TensorTraind::compressCI, "reltol"_a=1e-12, "maxBondDim"_a=0)
            .def("trueError",&TensorTraind::trueError, "f"_a, "max_n_eval"_a=1000000)
            .def("save", py::overload_cast<string>(&TensorTraind::save, py::const_))
            .def_static("load", py::overload_cast<string>(&TensorTraind::load))
            ;

    m.def("sum", xfac::operator+<T>,"tt1"_a, "tt2"_a);
    m.def("sum", &xfac::sum<T>, "tts"_a, "reltol"_a=1e-12, "maxBondDim"_a=0, "use_svd"_a=false);

    using QTensorTraind=QTensorTrain<T>;
    py::class_<QTensorTraind>(m, ("QTensorTrain"s+typestr).c_str())
            .def(py::init<TensorTraind, grid::Quantics>(), "tt"_a,"grid"_a)
            .def_readonly("tt",  &QTensorTraind::tt)
            .def_readonly("grid",  &QTensorTraind::grid)
            .def("eval", &QTensorTraind::eval)
            .def("integral", &QTensorTraind::integral)
            .def("save", py::overload_cast<string>(&QTensorTraind::save, py::const_))
            .def_static("load", py::overload_cast<string>(&QTensorTraind::load))
            ;

    using tensorF=function<T(vector<int>)>;
    using TensorCI1d=TensorCI1<T>;

    py::class_<TensorCI1d>(m, ("TensorCI1"s+typestr).c_str())
            .def_readwrite("param", &TensorCI1d::param)
            .def_readwrite("pivotError", &TensorCI1d::pivotError)
            .def("getIset", [](TensorCI1d const& ci) { return multiIndex_as_vec(ci.Iset); })
            .def("getLocalSet", [](TensorCI1d const& ci) { return multiIndex_as_vec(ci.localSet); })
            .def("getJset", [](TensorCI1d const& ci) { return multiIndex_as_vec(ci.Jset); })
            .def_readonly("T3", &TensorCI1d::T3)
            .def_readonly("P", &TensorCI1d::P)
            .def_readonly("cIter", &TensorCI1d::cIter)
            .def(py::init<tensorF,vector<int>,TensorCI1Param>(), "f"_a, "localDim"_a, "param"_a=TensorCI1Param())
            .def("iterate",  &TensorCI1d::iterate, "nIter"_a=1)
            .def("addPivotAt",  &TensorCI1d::addPivotAt)
            .def("len",  &TensorCI1d::len)
            .def("trueError",  &TensorCI1d::trueError, "max_n_eval"_a=1e6)
            .def("get_TensorTrain",  &TensorCI1d::get_TensorTrain, "center"_a=-1)
            .def("getPivotAt",  &TensorCI1d::getPivotsAt, "bond"_a)
            ;

    using TensorCI2d=TensorCI2<T>;

    py::class_<TensorCI2d>(m, ("TensorCI2"s+typestr).c_str())
            .def_readwrite("param", &TensorCI2d::param)
            .def_readwrite("pivotError", &TensorCI2d::pivotError)
            .def("getIset", [](TensorCI2d const& ci) { return multiIndex_as_vec(ci.Iset); })
            .def("getLocalSet", [](TensorCI2d const& ci) { return multiIndex_as_vec(ci.localSet); })
            .def("getJset", [](TensorCI2d const& ci) { return multiIndex_as_vec(ci.Jset); })
            .def_readonly("tt", &TensorCI2d::tt, "the tensor train")
            .def_readonly("P", &TensorCI2d::P, "the pivot matrix in LU form for each bond")
            .def_readonly("cIter", &TensorCI2d::cIter)
            .def_readonly("center", &TensorCI2d::center)
            .def(py::init<tensorF,vector<int>,TensorCI2Param>(), "f"_a, "localDim"_a, "param"_a=TensorCI2Param())
            .def(py::init<tensorF,TensorTraind,TensorCI2Param>(), "f"_a, "tt"_a, "param"_a=TensorCI2Param())
            .def(py::init<TensorTraind,TensorCI2Param>(), "tt"_a, "param"_a=TensorCI2Param())
            .def("addPivotsAllBonds",  &TensorCI2d::addPivotsAllBonds)
            .def("addPivotsAt", &TensorCI2d::addPivotsAt)
            .def("addPivots", &TensorCI2d::template addPivots<TensorCI2d>)
            .def("makeCanonical",  &TensorCI2d::makeCanonical)
            .def("iterate",  &TensorCI2d::iterate, "nIter"_a=1, "dmrg_type"_a=2)
            .def("len",  &TensorCI2d::len)
            .def("trueError",  &TensorCI2d::trueError, "max_n_eval"_a=1000000)
            .def("getPivotsAt",  &TensorCI2d::getPivotsAt, "bond"_a)
            .def("isDone", &TensorCI2d::isDone)
            ;


    using CTensorTraind=CTensorTrain<T,double>;
    py::class_<CTensorTraind>(m, ("CTensorTrain"s+typestr).c_str())
            .def_readonly("core",  &CTensorTraind::M)
            .def("eval", &CTensorTraind::eval)
            ;

    using ctensorF=function<T(vector<double>)>;
    using CTensorCI1d=CTensorCI1<T,double>;

    py::class_<CTensorCI1d,TensorCI1d>(m, ("CTensorCI1"s+typestr).c_str())
            .def(py::init<ctensorF,vector<vector<double>>,TensorCI1Param>(), "f"_a, "xi"_a, "args"_a=TensorCI1Param())
            .def("get_CTensorTrain", &CTensorCI1d::get_CTensorTrain)
            .def("get_T_at", &CTensorCI1d::get_T_at)
            ;

    using CTensorCI2d=CTensorCI2<T,double>;

    py::class_<CTensorCI2d,TensorCI2d>(m, ("CTensorCI2"s+typestr).c_str())
            .def(py::init<ctensorF,vector<vector<double>>,TensorCI2Param>(), "f"_a, "xi"_a, "args"_a=TensorCI2Param())
            .def("get_CTensorTrain", &CTensorCI2d::get_CTensorTrain)
            .def("get_T_at", &CTensorCI2d::get_T_at, "bond"_a, "x"_a)
            .def("get_TP1_at", &CTensorCI2d::get_TP1_at, "bond"_a, "x"_a)
            .def("get_P1T_at", &CTensorCI2d::get_P1T_at, "bond"_a, "x"_a)
            ;

    using QTensorCId=QTensorCI<T>;

    py::class_<QTensorCId,TensorCI2d>(m, ("QTensorCI"s+typestr).c_str())
            .def(py::init<ctensorF,grid::Quantics,TensorCI2Param>(), "f"_a, "qgrid"_a, "args"_a=TensorCI2Param())
            .def(py::init<function<T(double)>,grid::Quantics,TensorCI2Param>(), "f1d"_a, "qgrid"_a, "args"_a=TensorCI2Param())
            .def_readonly("qgrid", &QTensorCId::grid)
            .def("get_qtt", &QTensorCId::get_qtt)
            .def("addPivotPoints", &QTensorCId::addPivotPoints)
            .def("addPivotValues", &QTensorCId::addPivotValues)
            ;


    // tci1 <---> tci2 conversion
    m.def("to_tci1", &xfac::to_tci1<T>, "tci2"_a);
    m.def("to_tci2", py::overload_cast<const TensorCI1d&,tensorF,TensorCI2Param>(&xfac::to_tci2<T>), "tci1"_a, "g"_a, "param"_a);
    m.def("to_tci2", py::overload_cast<const TensorCI1d&,tensorF>(&xfac::to_tci2<T>), "tci1"_a, "g"_a);
    m.def("to_tci2", py::overload_cast<const TensorCI1d&,TensorCI2Param>(&xfac::to_tci2<T>), "tci1"_a, "param"_a);
    m.def("to_tci2", py::overload_cast<const TensorCI1d&>(&xfac::to_tci2<T>), "tci1"_a);
}


PYBIND11_MODULE(xfacpy, m) {
    m.doc() = "Python interface for tensor train cross interpolation (xfac)";

    /// can be converted directly to python list
    py::class_<MultiIndex>(m, "MultiIndex")
            .def("__iter__", [](MultiIndex &v) {
        return py::make_iterator(v.begin(), v.end());
    }, py::keep_alive<0, 1>()) /* Keep vector alive while iterator is used */
            ;



    py::class_<TensorCI1Param>(m,"TensorCI1Param")
            .def(py::init<>())
            .def_readwrite("nIter",&TensorCI1Param::nIter)
            .def_readwrite("reltol",&TensorCI1Param::reltol)
            .def_readwrite("pivot1",&TensorCI1Param::pivot1)
            .def_readwrite("fullPiv",&TensorCI1Param::fullPiv)
            .def_readwrite("nRookIter",&TensorCI1Param::nRookIter)
            .def_readwrite("weight",&TensorCI1Param::weight)
            .def_readwrite("cond",&TensorCI1Param::cond)
            .def_readwrite("useCachedFunction",&TensorCI1Param::useCachedFunction)
            ;

    py::class_<TensorCI2Param>(m,"TensorCI2Param")
            .def(py::init<>())
            .def_readwrite("bondDim",&TensorCI2Param::bondDim)
            .def_readwrite("reltol",&TensorCI2Param::reltol)
            .def_readwrite("pivot1",&TensorCI2Param::pivot1)
            .def_readwrite("fullPiv",&TensorCI2Param::fullPiv)
            .def_readwrite("nRookIter",&TensorCI2Param::nRookIter)
            .def_readwrite("weight",&TensorCI2Param::weight)
            .def_readwrite("cond",&TensorCI2Param::cond)
            .def_readwrite("useCachedFunction",&TensorCI2Param::useCachedFunction)
            ;

    declare_TensorCI<double>(m,"");
    declare_TensorCI<cmpx>(m,"_complex");

    m.def("GK15",&grid::QuadratureGK15,"a"_a=0,"b"_a=1);

    using grid::Quantics;
    py::class_<Quantics>(m,"QuanticsGrid")
            .def(py::init<double,double,int,int,bool>(), "a"_a=0,"b"_a=1,"nBit"_a=10,"dim"_a=1,"fused"_a=false)
            .def_readonly("deltaX",&Quantics::deltaX)
            .def_readonly("deltaVolume",&Quantics::deltaVolume)
            .def_readonly("tensorLen",&Quantics::tensorLen)
            .def_readonly("tensorLocDim",&Quantics::tensorLocDim)
            .def("coord_to_id", &Quantics::coord_to_id)
            .def("id_to_coord", &Quantics::id_to_coord)
            ;

    m.def("funtestRn",[](const vector<double>& xs)
    {
        double x=0, y=0,c=0;
        for(auto xi:xs) {c++; x+=c*xi; y+=xi*xi/c;}
        double arg=1.0+(x+2*y+x*y)*M_PI;
        return 1+x+cos(arg)+x*x+0.5*sin(arg);
    });

}

