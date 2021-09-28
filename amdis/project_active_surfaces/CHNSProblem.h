//
// Created by claudia on 17.09.2021.
//

#include "amdis/extensions/BaseProblem.hpp"


using namespace Dune::Functions::BasisFactory;
namespace AMDiS
{
    template <class Grid>
    //auto stokesBasis = composite(power<2>(lagrange<2>()), lagrange<1>());
    //auto chBasis = power<2>(lagrange<2>());
    //auto concentrBasis = lagrange<2>();

    //auto  CHNSTraits = composite(chBasis, stokesBasis, concentrBasis);
           // using CHNSTraits = LagrangeBasis<Grid, 1,1>; //TODO: Define correct basis

    template <class Grid>
            class CHNSProblem
                    : public BaseProblem<CHNSTraits<Grid>>
                    {
                using Super = BaseProblem<CHNSTraits<Grid>>;

                    public:
                        using GlobalBasis = typename Super::GlobalBasis;

                    public:
                        CHNSProblem(std::string const& name, Grid& grid, int n = 0);

                        void initTimestep(AdaptInfo& adaptInfo) override;

                        void initData(AdaptInfo& adaptInfo) override;

                        void fillCahnHilliard(AdaptInfo& adaptInfo) override;

                        void fillNavierStokes(AdaptInfo& adaptInfo) override;

                        void fillOperators(AdaptInfo& adaptInfo) override;

                        void fillBoundaryConditions(AdaptInfo& adaptInfo) override;

                        auto energy() const;

                        // bulk free-energy density
                        auto f() const
                        {
                            auto phi = getPhase();
                            return pow<2>(phi)*pow<2>(1.0 - phi);
                        }

                        // Mobility
                        auto M() const
                        {
                            return sot(max(1.e-7, f()), 4);
                        }

                        auto getPhase(int = 0)       { return this->solution(0);  }
                        auto getPhase(int = 0) const { return this->solution(0);  }

                        auto getOldPhase(int = 0)       { return valueOf(*oldSolution_,0); }
                        auto getOldPhase(int = 0) const { return valueOf(*oldSolution_,0); }

                        auto getW(int = 0)       { return this->solution(1); }
                        auto getW(int = 0) const { return this->solution(1); }

                        using Super::problem;

                    protected:
                        int n_ = 0;
                        double eps_ = 0.1;
                        double sigma_ = 1.0;

                        std::shared_ptr<DOFVector<GlobalBasis>> oldSolution_;
                    };

    // deduction guide
    template <class Grid, class BasisFactory>
            CHNSProblem (std::string name, Grid& grid, const BasisFactory& bf)
            -> CHNSProblem <Impl::DeducedProblemTraits_t<Grid,BasisFactory>>;

} // end namespace AMDiS

#include "amdis/project_active_surfaces/CHNSProblem.impl.h"
