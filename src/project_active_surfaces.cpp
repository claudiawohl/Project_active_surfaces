#include <amdis/AMDiS.hpp>
#include <amdis/AdaptInstationary.hpp>
#include <amdis/LocalOperators.hpp>
#include <amdis/ProblemInstat.hpp>
#include <amdis/ProblemStat.hpp>
#include <amdis/GridFunctions.hpp>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <amdis/Marker.hpp>

#include <amdis/localoperators/StokesOperator.hpp>

#include <dune/grid/albertagrid.hh>
//#include <dune/alugrid/grid.hh>
//#include <dune/foamgrid/foamgrid.hh>

using namespace AMDiS;
using namespace Dune::Functions::BasisFactory;

using Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
//using Param = LagrangeBasis<Grid, 1, 1, 2, 2, 2, 1>;

//Dune::YaspGrid<2> Grid();
//auto Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
//auto Grid = Dune::ALUGrid<GRIDDIM, WORLDDIM, Dune::simplex, Dune::nonconforming>();
//auto Grid = Dune::FoamGrid<GRIDDIM, WORLDDIM>();
//Dune::FoamGrid<GRIDDIM, WORLDDIM> Grid();

auto _0 = Dune::Indices::_0;
auto _1 = Dune::Indices::_1;

int main(int argc, char** argv)
{
    Environment env(argc, argv);

    auto grid = MeshCreator<Grid>("chMesh").create();

    auto stokesBasis = composite(power<WORLDDIM>(lagrange<2>()), lagrange<1>());
    auto chBasis = power<2>(lagrange<1>());

    auto chnsBasis = composite(chBasis, stokesBasis);

    ProblemStat prob("ch", *grid, chnsBasis);
    prob.initialize(INIT_ALL);

    ProblemInstat probInstat("ch", prob);
    probInstat.initialize(INIT_UH_OLD);

    AdaptInfo adaptInfo("adapt");

    auto invTau = std::ref(probInstat.invTau());
    auto phi = prob.solution(_0,0);
    auto phiOld = probInstat.oldSolution(_0,0);
    auto gradPhi = gradientOf(phi);

    auto _phi =makeTreePath(_0,0);
    auto _mu =makeTreePath(_0,1);
    auto _v =makeTreePath(_1,_0);
    auto _p =makeTreePath(_1,1);

    auto grav_force = [](auto const& x)
    {
        return FieldVector<double,  WORLDDIM>{0.0, 0.98};
    };
    auto opFconv = makeOperator(tag::test_trialvec{}, gradPhi);

    prob.addMatrixOperator(zot(invTau), _phi, _phi);
    prob.addMatrixOperator(opFconv, _phi, _v);
    prob.addVectorOperator(zot(phiOld * invTau), _phi);

    double M = Parameters::get<double>("parameters->mobility").value_or(1.0);
    prob.addMatrixOperator(sot(M), _phi, _mu);
    prob.addMatrixOperator(zot(1.0), _mu, _mu);

    double a = Parameters::get<double>("parameters->a").value_or(1.0);
    double b = Parameters::get<double>("parameters->b").value_or(1.0/4.0);

    double eps = Parameters::get<double>("parameters->epsilon").value_or(0.02);
    prob.addMatrixOperator(sot(-a*eps), _mu, _phi);

    //double Well potential: phi^2(1-phi)^2
    auto opFimpl = zot(-b/eps * (2 + 12*phi*(phi - 1)));
    prob.addMatrixOperator(opFimpl, _mu, _phi);

    auto opFexpl = zot(b/eps * pow<2>(phi)*(6 - 8*phi));
    prob.addVectorOperator(opFexpl, _mu);

    /*
    double viscosity = 1;
    auto opStokes = makeOperator(tag::stokes{0}, viscosity);
    prob.addMatrixOperator(opStokes, _1, _1);*/

    // define a constant fluid density
    double density_inner = 100.0;
    double density_outer =1000.0;

// <1/tau * u, v>
    auto opTime = makeOperator(tag::testvec_trialvec{},
                               (phi*density_inner + (1-phi)*density_outer) * invTau);
    prob.addMatrixOperator(opTime, _v, _v);

// <1/tau * u^old, v>
    auto opTimeOld = makeOperator(tag::testvec{},
                                  (phi*density_inner + (1-phi)*density_outer)* invTau * probInstat.oldSolution(_v));
    prob.addVectorOperator(opTimeOld, _v);

    for (int i = 0; i < Grid::dimensionworld; ++i) {
        // <(u^old * nabla)u_i, v_i>
        auto opNonlin = makeOperator(tag::test_gradtrial{},
                                     (phi*density_inner + (1-phi)*density_outer) * prob.solution(_v));
        prob.addMatrixOperator(opNonlin, makeTreePath(_1, _0, i), makeTreePath(_1, _0, i));
    }

// define  a fluid viscosity
    double viscosity = 1.0;
    double outer_visc = 10.;
    double inner_visc = 1.;

    for (int i = 0; i < Grid::dimensionworld; ++i) {
        // <viscosity*grad(u_i), grad(v_i)>
        auto opL = sot(inner_visc*phi+outer_visc*(1-phi));
        prob.addMatrixOperator(opL, makeTreePath(_1, _0, i), makeTreePath(_1, _0, i));
    }

    // <d_i(v_i), p>
    auto opDiv = makeOperator(tag::test_divtrialvec{}, 1.0);
    prob.addMatrixOperator(opDiv, makeTreePath(_1,_1), makeTreePath(_1,_0));

    // <q, d_i(u_i)>
    auto opP = makeOperator(tag::divtestvec_trial{}, 1.0);
    prob.addMatrixOperator(opP, makeTreePath(_1,_0), makeTreePath(_1,_1));

    double sigma = 24.5;
    for (int i=0; i<WORLDDIM; i++){
        auto partPhiOld = partialDerivativeOf(phi,i);
        prob.addMatrixOperator(zot(-sigma*partPhiOld), makeTreePath(_1,_0, i), _mu);
    }

/*    auto opForce = makeOperator(tag::testvec{}, grav_force, 0);
    prob.addVectorOperator(opForce, _v);*/

    int ref_int  = Parameters::get<int>("refinement->interface").value_or(10);
    int ref_bulk = Parameters::get<int>("refinement->bulk").value_or(2);
    GridFunctionMarker marker("interface", prob.grid(),
                              invokeAtQP([ref_int, ref_bulk](double phi) {
                                  return phi > 0.05 && phi < 0.95 ? ref_int : ref_bulk;
                              }, phi));
    prob.addMarker(marker);

    double radius1 = Parameters::get<double>("parameters->radius1").value_or(0.15);
    double radius2 = Parameters::get<double>("parameters->radius2").value_or(0.25);
    for (int i = 0; i < 6; ++i) {
        phi << [eps,radius1,radius2](auto const& x) {
            using Math::sqr;
            return 0.5*(1 - std::tanh((radius1+radius2)*(std::sqrt(sqr((x[0])/radius1) + sqr((x[1])/radius2)) - 1.0)/(4*std::sqrt(2.0)*eps)))
                   +0.5*(1 - std::tanh((0.1+0.3)       *(std::sqrt(sqr((x[0]+0.6)/0.1) + sqr((x[1]-0.5)/0.3)) - 1.0)/(4*std::sqrt(2.0)*eps)));
        };
        prob.markElements(adaptInfo);
        prob.adaptGrid(adaptInfo);
    }

    AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
    adapt.adapt();

    return 0;
}
