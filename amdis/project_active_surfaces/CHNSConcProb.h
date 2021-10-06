//
// Created by claudia on 06.10.2021.
//

include <amdis/extensions/BaseProblem.hpp>
#include <amdis/extensions/CouplingBaseProblem.hpp>

#include "CahnHilliard.hpp"
#include "NavierStokes.hpp"

namespace AMDiS
{
template <class Traits>
class CHNSConcProb
        : public CouplingBaseProblem
        {
    using Self = CHNSConc;
    using Super = CouplingBaseProblem;

    using CHNS = typename Traits::CahnHilliardNavierStokes;
    using Conc = typename Traits::Concentration;

        public:
            using GridView = typename Traits::GridView;
            using Grid = typename GridView::Grid;

            enum { dow = Grid::dimensionworld };
            enum { dim = Grid::dimension };

        public:
            CHNSConcProb(Grid& grid);
            : Super("chnsconc", std::make_shared<CHNS>("chns",grid), std::make_shared<Conc>("conc",grid,1))
            , chnsProb_(std::dynamic_pointer_cast<CHNS>(Super::baseProblem(0)))
            , concProb_(std::dynamic_pointer_cast<Conc>(Super::baseProblem(1)))
            {
                Parameters::get("parameters->eps", eps_);
                Parameters::get("parameters->sigma", sigma_);
            }

            void fillCouplingOperators(AdaptInfo &) override {
                chnsProb_.addMatrixoperator(zot(1.), 0, 0)
            }


        }
}