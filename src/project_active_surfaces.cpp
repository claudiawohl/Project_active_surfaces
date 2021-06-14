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
#include <cmath>
#include <typeinfo>
//#include <dune/alugrid/grid.hh>
//#include <dune/foamgrid/foamgrid.hh>

using namespace AMDiS;
using namespace Dune::Functions::BasisFactory;

using Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
//using Param = LagrangeBasis<Grid, 1, 1, 2, 2, 2, 1>;
//using Grid = Dune::YaspGrid<WORLDDIM>;


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
    //Dune::YaspGrid<2> grid{ {1.,2.}, {8,8}};

    auto stokesBasis = composite(power<WORLDDIM>(lagrange<2>()), lagrange<1>());
    auto chBasis = power<2>(lagrange<2>());

    auto chnsBasis = composite(chBasis, stokesBasis);

    ProblemStat prob("ch", *grid, chnsBasis);
    prob.initialize(INIT_ALL);

    ProblemInstat probInstat("ch", prob);
    probInstat.initialize(INIT_UH_OLD);

    AdaptInfo adaptInfo("adapt");

    //FUNCTIONS from the previous time step
    auto invTau = std::ref(probInstat.invTau());
    auto phi = prob.solution(_0,0);
//    auto phiOld = probInstat.oldSolution(_0,0);  //replaced by phi
    auto gradPhi = gradientOf(phi);
    auto v = prob.solution(_1,_0);

    auto upper_bound = [](auto const& x){
        return Dune::FieldVector<double,1>{1.};
    };
    auto lower_bound = [](auto const& x) {
        return Dune::FieldVector<double, 1>{0.};
    };
    auto phiProjected = max(min(evalAtQP(phi),upper_bound),lower_bound);
    //auto phiProjected = phi;

    //PATHS
    auto _phi =makeTreePath(_0,0);
    auto _mu =makeTreePath(_0,1);
    auto _v =makeTreePath(_1,_0);
    auto _p =makeTreePath(_1,1);

    //FIRST EQUATION
    //time derivative
    prob.addMatrixOperator(zot(invTau), _phi, _phi);
    prob.addVectorOperator(zot(phi * invTau), _phi);

    //Coupling term
    //auto opFconv = makeOperator(tag::test_gradtrial{}, v);
    //prob.addMatrixOperator(opFconv, _phi, _phi);

    //auto opFconv = makeOperator(tag::test_gradtrial{}, FieldVector<double, 2>{0, 0.98});
    //prob.addMatrixOperator(opFconv, _phi, _phi);

    auto opFconv = makeOperator(tag::test_trialvec{}, gradPhi, 5);
    prob.addMatrixOperator(opFconv, _phi, _v);                          //Coupling term

    //mobility term
    double m = Parameters::get<double>("parameters->mobility").value_or(0.4);
    prob.addMatrixOperator(sot(m*pow<2>(phi)*pow<2>(FieldVector<double,1>{1.}-phi)), _phi, _mu);

    //SECOND EQUATION
    prob.addMatrixOperator(zot(1.0), _mu, _mu);

    double sigma = Parameters::get<double>("parameters->sigma").value_or(24.5)*3*sqrt(2);
    double a = Parameters::get<double>("parameters->a").value_or(1.0);
    double b = Parameters::get<double>("parameters->b").value_or(1.0);

    double eps = Parameters::get<double>("parameters->epsilon").value_or(0.02);
    prob.addMatrixOperator(sot(-a*eps), _mu, _phi);

    //double Well potential: phi^2(1-phi)^2
    //TODO: Change expression as well to vector? - Not necessary - why not?
    auto opFimpl = zot(-b/eps * (2 + 12*phi*(phi - 1)));
    prob.addMatrixOperator(opFimpl, _mu, _phi);

    auto opFexpl = zot(b/eps * pow<2>(phi)*(6 - 8*phi));
    prob.addVectorOperator(opFexpl, _mu);

    //THIRD EQUATION

    // define a constant fluid density
    double density_inner = 100.0;
    double density_outer = 1000.0;
    auto density = density_inner*phiProjected+(Dune::FieldVector<double,1>{1.}-phiProjected)*density_outer;


// <1/tau * u, v>
    auto opTime = makeOperator(tag::testvec_trialvec{},
                               density* invTau, 5);
    prob.addMatrixOperator(opTime, _v, _v);

// <1/tau * u^old, v>
    auto opTimeOld = makeOperator(tag::testvec{},
                                  density * invTau * prob.solution(_v), 5);
    prob.addVectorOperator(opTimeOld, _v);


     for (int i = 0; i < Grid::dimensionworld; ++i) {
        // <(u^old * nabla)u_i, v_i>
        auto opNonlin = makeOperator(tag::test_gradtrial{},
                                     density * prob.solution(_v), 5);
        prob.addMatrixOperator(opNonlin, makeTreePath(_1, _0, i), makeTreePath(_1, _0, i));
    }


// define  a fluid viscosity
    double outer_visc = 10.;
    double inner_visc = 1.;

    auto viscosity = inner_visc*phiProjected+outer_visc*(FieldVector<double,1>{1.}-phiProjected);

  //Laplace term
    /*
    for (int i = 0; i < Grid::dimensionworld; ++i) {
        // <viscosity*grad(u_i), grad(v_i)>
        auto opL = sot(viscosity, 1);
        prob.addMatrixOperator(opL, makeTreePath(_1, _0, i), makeTreePath(_1, _0, i));
    }
     */

    auto opVLaplace = makeOperator(tag::divtestvec_divtrialvec{},viscosity, 5);
    prob.addMatrixOperator(opVLaplace, _v, _v);

   // <q, d_i(u_i)>
    auto opDiv = makeOperator(tag::test_divtrialvec{}, 1.0);
   prob.addMatrixOperator(opDiv, makeTreePath(_1,_1), makeTreePath(_1,_0));

   /*
   auto opV= makeOperator(tag::testvec_trialvec{}, 1.);
   prob.addMatrixOperator(opV, makeTreePath(_1,_0), makeTreePath(_1,_0));

    auto opP= makeOperator(tag::test_trial{}, 1.);
    prob.addMatrixOperator(opP, makeTreePath(_1,_1), makeTreePath(_1,_1));*/

    // <d_i(v_i), p>
    auto opP = makeOperator(tag::divtestvec_trial{}, 1.0);
    prob.addMatrixOperator(opP, makeTreePath(_1,_0), makeTreePath(_1,_1));

    //coupling term
/*
    for (int i=0; i<WORLDDIM; i++){
        auto partphi = partialDerivativeOf(phi,i);
        prob.addMatrixOperator(zot(-sigma*partphi), makeTreePath(_1,_0, i), _mu);
    }
*/
    auto opCoup = makeOperator(tag::testvec_trial{}, -sigma*gradPhi);
    prob.addMatrixOperator(opCoup, _v, _mu);

    //extern force (gravitational force)
    auto extForce = [](auto const& x) { return FieldVector<double,  WORLDDIM>{0.0,- 0.98};};
    auto opForce = makeOperator(tag::testvec {}, density*extForce, 5);
    prob.addVectorOperator(opForce, _v);


    //Initial Value and Boundary
    int ref_int  = Parameters::get<int>("refinement->interface").value_or(10);
    int ref_bulk = Parameters::get<int>("refinement->bulk").value_or(2);
    GridFunctionMarker marker("interface", prob.grid(),
                              invokeAtQP([ref_int, ref_bulk](double phi) {
                                  return phi > 0.05 && phi < 0.95 ? ref_int : ref_bulk;
                              }, phi));
    prob.addMarker(marker);

    double radius = Parameters::get<double>("parameters->radius").value_or(0.5);
    double radius1 = Parameters::get<double>("parameters->radius1").value_or(0.15);
    double radius2 = Parameters::get<double>("parameters->radius2").value_or(0.25);

    //set initial value for Benchmark
    radius1 = radius;
    radius2 = radius;

    double centerx = Parameters::get<double>("parameters->centerx").value_or(0.);
    double centery = Parameters::get<double>("parameters->centery").value_or(0.);

    for (int i = 0; i < 6; ++i) {
        phi << [eps,radius1,radius2, centerx, centery](auto const& x) {
            using Math::sqr;
            return 0.5*(1 - std::tanh((radius1+radius2)*(std::sqrt(sqr((x[0]-centerx)/radius1) + sqr((x[1]-centery)/radius2)) - 1.0)/(4*std::sqrt(2.0)*eps)));
        };
        prob.markElements(adaptInfo);
        prob.adaptGrid(adaptInfo);
    }

    //boundary condition
    prob.addDirichletBC([](auto const& x) {return (x[1]<1.e-8 || x[1]>2-1.e-8);}, _v, _v, FieldVector<double, WORLDDIM>{0., 0.});
    prob.addDirichletBC([](auto const& x) {return (x[0]<1.e-8 || x[0]>1-1.e-8);}, makeTreePath(_1,_0,0), makeTreePath(_1,_0,0), 0.);

    AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
    adapt.adapt();

    return 0;
}