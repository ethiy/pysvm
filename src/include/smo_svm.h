#pragma once

#include <cstddef>
#include <numeric_limits>

#include <map>
#include <deque>

#include <boost/optional.hpp>

namespace svm
{
    template<typename Kernel>
    class BinarySMOSVM
    {
    public:
        BinarySMOSVM(void);
        BinarySMOSVM(
            std::size_t max_iterations = std::numeric_limits<std::size_t>::infinity(),
            Kernel kernel,
        )
        BinarySMOSVM(BinarySMOSVM const& other) = delete;
        BinarySMOSVM(BinarySMOSVM && other);
        ~BinarySMOSVM(void);

        void swap(BinarySMOSVM & other);

        BinarySMOSVM & operator =(BinarySMOSVM const& other) = delete;
        BinarySMOSVM & operator =(BinarySMOSVM && other);

        BinarySMOSVM & fit<typename Instance_Type>(std::vector<Instance_Type> const& instances, std::vector<short> const& classes);
        short predict<typename Instance_Type>(Instance_Type const& instance);
        double score<typename Instance_Type>(boost::optional< std::vector<Instance_Type> > const& instances = boost::none, boost::optional< std::vector<short> > const& classes = boost::none);
    private:
        std::size_t max_iterations = 1000;
        Kernel kernel;
        double C = 1;
        double tolerance = 1E-3;

        bool debug = false;
        bool verbose = false;

        double b = 0;
        double b_up = -1;
        double b_low = 1;

        std::size_t i_up = 0;
        std::size_t i_low = 0;

        double L = 0;
        double H = 0;
        double delta_i = 0;
        double delta_j = 0;

        std::map<std::pair<std::size_t, std::size_t>, double> eta;

        std::deque<std::size_t> support_verctor_indices;

        std::size_t updated = 0;
        bool visit_all = true;

        std::list<double> alpha_updates{{std::numeric_limits<double>::infinity()}};
        std::list<double> lds{{0}};
    };
}
