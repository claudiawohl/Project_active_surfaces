///@author Simon Praetorius

#include <amdis/AMDiS.hpp>
#include <amdis/AdaptInstationary.hpp>
#include <amdis/LocalOperators.hpp>
#include <amdis/ProblemInstat.hpp>
#include <amdis/ProblemStat.hpp>
#include <amdis/GridFunctions.hpp>
#include <amdis/Marker.hpp>

#include <dune/grid/albertagrid.hh>

using namespace AMDiS;

using Grid = Dune::AlbertaGrid<GRIDDIM, WORLDDIM>;
using Param = LagrangeBasis<Grid, 1, 1>;

int main(int argc, char** argv)
{
  Environment env(argc, argv);

  ProblemStat<Param> prob("ch");
  prob.initialize(INIT_ALL);

  ProblemInstat<Param> probInstat("ch", prob);
  probInstat.initialize(INIT_UH_OLD);

  AdaptInfo adaptInfo("adapt");

  auto invTau = std::ref(probInstat.invTau());
  auto phi = prob.solution(0);
  auto phiOld = probInstat.oldSolution(0);

  prob.addMatrixOperator(zot(invTau), 0, 0);
  prob.addVectorOperator(zot(phiOld * invTau), 0);

  double M = Parameters::get<double>("parameters->mobility").value_or(1.0);
  prob.addMatrixOperator(sot(M), 0, 1);
  if (WORLDDIM == 3)
    {
        auto extra3D = [M](auto const& x){
            return M/x[0];
        };

        auto opRemoveAngle = makeOperator(tag::partialtest_partialtrial {1,1}, M);
        prob.addMatrixOperator(opRemoveAngle, 0, 1);

        auto opAddExtra = makeOperator(tag::test_partialtrial{0}, extra3D);
        prob.addMatrixOperator(opAddExtra, 1, 0);
    };

  prob.addMatrixOperator(zot(1.0), 1, 1);

  double a = Parameters::get<double>("parameters->a").value_or(1.0);
  double b = Parameters::get<double>("parameters->b").value_or(1.0/4.0);

  double eps = Parameters::get<double>("parameters->epsilon").value_or(0.02);
  prob.addMatrixOperator(sot(-a*eps), 1, 0);
  if (WORLDDIM == 3)
  {
      auto extra3D = [a, eps](auto const& x){
          double factor = 1.;
          return -a*eps/x[0];
      };

    auto opRemoveAngle = makeOperator(tag::partialtest_partialtrial {1,1}, -a*eps);
    prob.addMatrixOperator(opRemoveAngle, 1, 0);

    auto opAddExtra = makeOperator(tag::test_partialtrial{0}, extra3D);
    prob.addMatrixOperator(opAddExtra, 1, 0);
  };

  auto opFimpl = zot(-b/eps * (2 + 12*phi*(phi - 1)));
  prob.addMatrixOperator(opFimpl, 1, 0);

  auto opFexpl = zot(b/eps * pow<2>(phi)*(6 - 8*phi));
  prob.addVectorOperator(opFexpl, 1);

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
      return 0.5*(1 - std::tanh((radius1+radius2)*(std::sqrt(sqr(x[0]/radius1) + sqr(x[1]/radius2)) - 1.0)/(4*std::sqrt(2.0)*eps)));
    };
    prob.markElements(adaptInfo);
    prob.adaptGrid(adaptInfo);
  }

  AdaptInstationary adapt("adapt", prob, adaptInfo, probInstat, adaptInfo);
  adapt.adapt();

  return 0;
}
