// Author: Aaryan Pradhan
// Financial Value Note and Delivery Contract Implementation


#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include <algorithm>
#include <random>
#include <memory>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <cassert>

namespace Task1 {

    /**
     * Hybrid root-finding algorithm combining bracketing and Newton-Raphson methods
     * @param f Function to find root of
     * @param df First derivative of function
     * @param d2f Second derivative of function
     * @param target Target value (finding x where f(x) = target)
     * @param x_lo Initial lower bound
     * @param x_hi Initial upper bound
     * @param tol Tolerance for convergence
     * @param maxBracketIters Maximum iterations for bracketing
     * @param maxPolishIters Maximum iterations for polishing
     * @return Root of the equation
     */
    double solveHybrid(
        const std::function<double(double)>& f,
        const std::function<double(double)>& df,
        const std::function<double(double)>& d2f,
        double target,
        double x_lo,
        double x_hi,
        double tol,
        int maxBracketIters,
        int maxPolishIters
    ) {
        // Transform to root-finding problem: g(x) = f(x) - target = 0
        auto g = [&](double x) { return f(x) - target; };
        
        // Bracketing phase: find interval where function changes sign
        double a = x_lo, b = x_hi;
        double fa = g(a), fb = g(b);
        
        for (int it = 0; it < maxBracketIters && fa * fb > 0.0; ++it) {
            double mid = 0.5 * (a + b);
            double w = b - a;
            a = std::max(0.0, mid - w);
            b = mid + w;
            fa = g(a);
            fb = g(b);
        }
        
        if (fa * fb > 0.0) {
            throw std::runtime_error("solveHybrid: could not bracket root");
        }
        
        // Bisection refinement: narrow down the interval
        for (int it = 0; it < 5; ++it) {
            double m = 0.5 * (a + b);
            double fm = g(m);
            if (fa * fm <= 0.0) {
                b = m;
                fb = fm;
            } else {
                a = m;
                fa = fm;
            }
        }
        
        // Newton-Raphson polishing phase with Halley's method modification
        double x = 0.5 * (a + b);
        for (int it = 0; it < maxPolishIters; ++it) {
            double gx = g(x);
            double d1x = df(x);
            double d2x = d2f(x);
            
            if (std::fabs(d1x) < 1e-16) break;
            
            // Modified Newton step (Halley's method for better convergence)
            double denom = d1x - 0.5 * gx * (d2x / d1x);
            if (std::fabs(denom) < 1e-16) break;
            
            double dx = gx / denom;
            x -= dx;
            
            if (std::fabs(dx) < tol) break;
        }
        
        return x;
    }

    /**
     * Abstract base class for Value Notes
     * Defines the interface for different pricing conventions
     */
    class ValueNote {
    public:
        double N;   // Notional amount
        double M;   // Maturity (years)
        double VR;  // Value rate (%)
        int PF;     // Payment frequency (per year)
        
        ValueNote(double n, double m, double vr, int pf)
            : N(n), M(m), VR(vr), PF(pf) {}
        
        virtual ~ValueNote() = default;
        
        // Pure virtual functions defining the pricing interface
        virtual double price_from_ER(double ER) const = 0;
        virtual double ER_from_price(double VP) const = 0;
        virtual double dVP_dER(double ER) const = 0;
        virtual double dER_dVP(double VP) const = 0;
        virtual double d2VP_dER2(double ER) const = 0;
        virtual double d2ER_dVP2(double VP) const = 0;
    };

    /**
     * Linear Convention Value Note
     * Simple linear relationship between price and effective rate
     */
    class ValueNoteLinear : public ValueNote {
    public:
        ValueNoteLinear(double n, double m, double vr, int pf)
            : ValueNote(n, m, vr, pf) {}
        
        double price_from_ER(double ER) const override {
            return N * (1.0 - ER * M / 100.0);
        }
        
        double ER_from_price(double VP) const override {
            return 100.0 * (1.0 - VP / N) / M;
        }
        
        double dVP_dER(double) const override {
            return -N * M / 100.0;
        }
        
        double dER_dVP(double) const override {
            return 1.0 / (-N * M / 100.0);
        }
        
        double d2VP_dER2(double) const override {
            return 0.0;  // Linear function has zero second derivative
        }
        
        double d2ER_dVP2(double) const override {
            return 0.0;  // Linear function has zero second derivative
        }
    };

    /**
     * Cumulative Convention Value Note
     * Standard bond pricing with periodic coupon payments
     */
    class ValueNoteCumulative : public ValueNote {
    public:
        ValueNoteCumulative(double n, double m, double vr, int pf)
            : ValueNote(n, m, vr, pf) {}
        
        int num_payments() const {
            return static_cast<int>(std::round(M * PF));
        }
        
        double payment_time(int i) const {
            return double(i + 1) / PF;
        }
        
        double coupon() const {
            return VR * N / (100.0 * PF);
        }
        
        double price_from_ER(double ER) const override {
            double u = 1.0 + ER / (100.0 * PF);
            int n = num_payments();
            double vp = 0.0;
            double c = coupon();
            
            // Present value of coupon payments
            for (int i = 0; i < n - 1; ++i) {
                double t = PF * payment_time(i);
                vp += c / std::pow(u, t);
            }
            
            // Present value of final payment (coupon + principal)
            double t = PF * payment_time(n - 1);
            vp += (c + N) / std::pow(u, t);
            
            return vp;
        }
        
        double dVP_dER(double ER) const override {
            double u = 1.0 + ER / (100.0 * PF);
            int n = num_payments();
            double sum = 0.0;
            double c = coupon();
            
            // Derivative of coupon payments
            for (int i = 0; i < n - 1; ++i) {
                double t = PF * payment_time(i);
                sum += -c * t / (100.0 * std::pow(u, t + 1));
            }
            
            // Derivative of final payment
            double tn = PF * payment_time(n - 1);
            sum += -(c + N) * tn / (100.0 * std::pow(u, tn + 1));
            
            return sum;
        }
        
        double d2VP_dER2(double ER) const override {
            double u = 1.0 + ER / (100.0 * PF);
            int n = num_payments();
            double sum = 0.0;
            double c = coupon();
            
            // Second derivative of coupon payments
            for (int i = 0; i < n - 1; ++i) {
                double t = PF * payment_time(i);
                sum += c * t * (t + 1) / (100.0 * 100.0 * std::pow(u, t + 2));
            }
            
            // Second derivative of final payment
            double tn = PF * payment_time(n - 1);
            sum += (c + N) * tn * (tn + 1) / (100.0 * 100.0 * std::pow(u, tn + 2));
            
            return sum;
        }
        
        double ER_from_price(double VP) const override {
            return solveHybrid(
                std::bind(&ValueNoteCumulative::price_from_ER, this, std::placeholders::_1),
                std::bind(&ValueNoteCumulative::dVP_dER, this, std::placeholders::_1),
                std::bind(&ValueNoteCumulative::d2VP_dER2, this, std::placeholders::_1),
                VP, 0.0, 100.0, 1e-12, 50, 20
            );
        }
        
        double dER_dVP(double VP) const override {
            double ER = ER_from_price(VP);
            return 1.0 / dVP_dER(ER);
        }
        
        double d2ER_dVP2(double VP) const override {
            double ER = ER_from_price(VP);
            double df = dVP_dER(ER);
            double d2f = d2VP_dER2(ER);
            return -d2f / (df * df * df);
        }
    };

    /**
     * Recursive Convention Value Note
     * Forward accumulation approach to bond pricing
     */
    class ValueNoteRecursive : public ValueNote {
    public:
        ValueNoteRecursive(double n, double m, double vr, int pf)
            : ValueNote(n, m, vr, pf) {}
        
        double price_from_ER(double ER) const override {
            int n = static_cast<int>(std::round(M * PF));
            double mi = 1.0 / PF;
            double c = VR * N / (100.0 * PF);
            double FV = 0.0;
            
            // Forward accumulation of coupon payments
            for (int i = 0; i < n; ++i) {
                FV = (FV + c) * (1.0 + ER * mi / 100.0);
            }
            
            // Discount back to present value
            return (N + FV) / (1.0 + ER * M / 100.0);
        }
        
        double dVP_dER(double ER) const override {
            int n = static_cast<int>(std::round(M * PF));
            double mi = 1.0 / PF;
            double c = VR * N / (100.0 * PF);
            double FV = 0.0;
            double dFV = 0.0;
            
            // Forward accumulation with derivative tracking
            for (int i = 0; i < n; ++i) {
                dFV = dFV * (1.0 + ER * mi / 100.0) + (FV + c) * (mi / 100.0);
                FV = (FV + c) * (1.0 + ER * mi / 100.0);
            }
            
            // Apply quotient rule for discounting
            double denom = 1.0 + ER * M / 100.0;
            double num = dFV * denom - (N + FV) * (M / 100.0);
            return num / (denom * denom);
        }
        
        double d2VP_dER2(double ER) const override {
            int n = static_cast<int>(std::round(M * PF));
            double mi = 1.0 / PF;
            double c = VR * N / (100.0 * PF);
            double FV = 0.0;
            double dFV = 0.0;
            double d2FV = 0.0;
            
            // Forward accumulation with first and second derivative tracking
            for (int i = 0; i < n; ++i) {
                d2FV = d2FV * (1.0 + ER * mi / 100.0) + 2.0 * dFV * (mi / 100.0);
                dFV = dFV * (1.0 + ER * mi / 100.0) + (FV + c) * (mi / 100.0);
                FV = (FV + c) * (1.0 + ER * mi / 100.0);
            }
            
            // Second derivative using quotient rule
            double denom = 1.0 + ER * M / 100.0;
            double num1 = d2FV * denom - 2.0 * (M / 100.0) * dFV;
            double num2 = 2.0 * (N + FV) * (M / 100.0) * (M / 100.0);
            return (num1 * denom - num2) / (denom * denom * denom);
        }
        
        double ER_from_price(double VP) const override {
            return solveHybrid(
                std::bind(&ValueNoteRecursive::price_from_ER, this, std::placeholders::_1),
                std::bind(&ValueNoteRecursive::dVP_dER, this, std::placeholders::_1),
                std::bind(&ValueNoteRecursive::d2VP_dER2, this, std::placeholders::_1),
                VP, 0.0, 100.0, 1e-12, 50, 20
            );
        }
        
        double dER_dVP(double VP) const override {
            double ER = ER_from_price(VP);
            return 1.0 / dVP_dER(ER);
        }
        
        double d2ER_dVP2(double VP) const override {
            double ER = ER_from_price(VP);
            double df = dVP_dER(ER);
            double d2f = d2VP_dER2(ER);
            return -d2f / (df * df * df);
        }
    };

    /**
     * Demonstration function for Task 1
     * Shows pricing and sensitivity calculations for all three conventions
     */
    void run() {
        double ER0 = 5.0, VP0 = 100.0;
        ValueNoteLinear vl(100, 5.0, 3.5, 1);
        ValueNoteCumulative vc(100, 5.0, 3.5, 1);
        ValueNoteRecursive vr(100, 5.0, 3.5, 1);

        std::cout << "=== Task 1: Conventions Demo ===\n";
        std::cout << std::fixed << std::setprecision(8);

        std::cout << "Linear   : price(5%) = " << vl.price_from_ER(ER0)
                  << ",  ER(100) = " << vl.ER_from_price(VP0) << "\n";
        std::cout << "Cum      : price(5%) = " << vc.price_from_ER(ER0)
                  << ",  ER(100) = " << vc.ER_from_price(VP0) << "\n";
        std::cout << "Rec      : price(5%) = " << vr.price_from_ER(ER0)
                  << ",  ER(100) = " << vr.ER_from_price(VP0) << "\n\n";

        std::cout << "dVP/dER @5%:\n"
                  << "  Lin: " << vl.dVP_dER(ER0) << "\n"
                  << "  Cum: " << vc.dVP_dER(ER0) << "\n"
                  << "  Rec: " << vr.dVP_dER(ER0) << "\n\n";

        std::cout << "dER/dVP @100:\n"
                  << "  Lin: " << vl.dER_dVP(VP0) << "\n"
                  << "  Cum: " << vc.dER_dVP(VP0) << "\n"
                  << "  Rec: " << vr.dER_dVP(VP0) << "\n\n";

        std::cout << "d2VP/dER2 @5%:\n"
                  << "  Lin: " << vl.d2VP_dER2(ER0) << "\n"
                  << "  Cum: " << vc.d2VP_dER2(ER0) << "\n"
                  << "  Rec: " << vr.d2VP_dER2(ER0) << "\n\n";

        std::cout << "d2ER/dVP2 @100:\n"
                  << "  Lin: " << vl.d2ER_dVP2(VP0) << "\n"
                  << "  Cum: " << vc.d2ER_dVP2(VP0) << "\n"
                  << "  Rec: " << vr.d2ER_dVP2(VP0) << "\n\n";
    }

} // namespace Task1

namespace Task2 {

    /**
     * Base class for Value Notes in Task 2
     * Focused on stochastic modeling and delivery contracts
     */
    class ValueNote {
    public:
        double notional;
        double maturity;
        double valueRate;
        double currentPrice;
        double currentEffectiveRate;
        double volatility;
        int paymentFrequency;
        
        ValueNote(double N, double M, double VR, int PF, double vol)
            : notional(N), maturity(M), valueRate(VR),
              paymentFrequency(PF), currentPrice(N),
              currentEffectiveRate(VR), volatility(vol) {}
        
        virtual ~ValueNote() = default;
        
        virtual double priceGivenYield(double effRate, double t = 0.0) const = 0;
        virtual double derivativePriceToYield(double effRate, double t = 0.0) const = 0;
        virtual double yieldGivenPrice(double price) const = 0;
    };

    /**
     * Cumulative Convention Value Note for Task 2
     * Enhanced with time-dependent pricing and derivatives
     */
    class ValueNoteCumulative : public ValueNote {
    public:
        ValueNoteCumulative(double N, double M, double VR, int PF, double vol)
            : ValueNote(N, M, VR, PF, vol) {
            currentEffectiveRate = VR;
            currentPrice = priceGivenYield(VR, 0.0);
        }
        
        double priceGivenYield(double effRate, double currentTime = 0.0) const override {
            if (currentTime > maturity) return 0.0;
            
            double y = effRate / 100.0;
            int PF = paymentFrequency;
            double start = currentTime;
            double totalPeriods = maturity * PF;
            int fullPeriods = int(std::floor(totalPeriods + 1e-12));
            bool partialFinal = (std::fabs(totalPeriods - fullPeriods) > 1e-9);
            double lastTime = maturity;
            int startPeriod = int(std::floor(start * PF + 1e-12)) + 1;
            
            double price = 0.0;
            double couponAmt = (valueRate / 100.0) * notional / PF;
            
            // Price coupon payments
            for (int j = startPeriod; j <= fullPeriods; ++j) {
                double payTime = j * (1.0 / PF);
                if (payTime >= lastTime - 1e-9) break;
                if (payTime <= start + 1e-12) continue;
                
                double t = payTime - start;
                price += couponAmt * std::pow(1 + y / PF, -PF * t);
            }
            
            // Price final payment (principal + final coupon)
            if (lastTime > start + 1e-9) {
                double t = lastTime - start;
                double finalC = partialFinal
                    ? (valueRate / 100.0) * notional * ((maturity - fullPeriods * (1.0 / PF)) * PF)
                    : couponAmt;
                double finalCash = notional + finalC;
                price += finalCash * std::pow(1 + y / PF, -PF * t);
            }
            
            return price;
        }
        
        double derivativePriceToYield(double effRate, double currentTime = 0.0) const override {
            if (currentTime > maturity) return 0.0;
            
            double y = effRate / 100.0;
            int PF = paymentFrequency;
            double start = currentTime;
            double totalPeriods = maturity * PF;
            int fullPeriods = int(std::floor(totalPeriods + 1e-12));
            bool partialFinal = (std::fabs(totalPeriods - fullPeriods) > 1e-9);
            double lastTime = maturity;
            int startPeriod = int(std::floor(start * PF + 1e-12)) + 1;
            
            double dPdE = 0.0;
            double couponAmt = (valueRate / 100.0) * notional / PF;
            
            // Derivative of coupon payments
            for (int j = startPeriod; j <= fullPeriods; ++j) {
                double payTime = j * (1.0 / PF);
                if (payTime >= lastTime - 1e-9) break;
                if (payTime <= start + 1e-12) continue;
                
                double t = payTime - start;
                double base = 1.0 + y / PF;
                dPdE += -couponAmt * t * std::pow(base, -PF * t - 1);
            }
            
            // Derivative of final payment
            if (lastTime > start + 1e-9) {
                double t = lastTime - start;
                double finalC = partialFinal
                    ? ((valueRate / 100.0) * notional * ((maturity - fullPeriods * (1.0 / PF)) * PF))
                    : couponAmt;
                double finalCash = notional + finalC;
                double base = 1.0 + y / PF;
                dPdE += -finalCash * t * std::pow(base, -PF * t - 1);
            }
            
            return dPdE / 100.0;
        }
        
        double yieldGivenPrice(double targetPrice) const override {
            double E = valueRate;
            double par = priceGivenYield(valueRate, 0.0);
            
            // Initial guess refinement
            if (targetPrice > par + 1e-6) E = valueRate * 0.5;
            if (targetPrice < par - 1e-6) E = valueRate * 1.5;
            
            double low = 0, high = 100;
            
            // Newton-Raphson with bisection fallback
            for (int i = 0; i < 50; ++i) {
                double f = priceGivenYield(E, 0.0) - targetPrice;
                double fp = derivativePriceToYield(E, 0.0);
                
                if (std::fabs(f) < 1e-8) break;
                
                if (std::fabs(fp) < 1e-12) {
                    E = 0.5 * (low + high);
                    continue;
                }
                
                double En = E - f / fp;
                if (En < low || En > high) En = 0.5 * (low + high);
                
                if (priceGivenYield(En, 0.0) > targetPrice) {
                    low = En;
                } else {
                    high = En;
                }
                
                if (std::fabs(En - E) < 1e-8) {
                    E = En;
                    break;
                }
                E = En;
            }
            
            return E;
        }
    };

    /**
     * Delivery Contract with advanced probability modeling
     * Handles basket of value notes with stochastic yield evolution
     */
    class DeliveryContract {
    private:
        std::vector<ValueNote*> basket;
        double SVR;  // Standard value rate
        double expiration;
        double riskFreeRate;
        std::vector<double> relativeFactors;
        
    public:
        std::vector<double> deliveryProb;
        std::vector<double> sensitivityVol;
        std::vector<double> sensitivityPrice;
        
        DeliveryContract(double svr, double expy, double rfr)
            : SVR(svr), expiration(expy), riskFreeRate(rfr) {}
        
        void addValueNote(ValueNote* vn) {
            basket.push_back(vn);
        }
        
        void calculateRelativeFactors() {
            relativeFactors.clear();
            for (auto* vn : basket) {
                relativeFactors.push_back(vn->priceGivenYield(SVR, 0.0) / 100.0);
            }
        }
        
        const std::vector<double>& getRelativeFactors() const {
            return relativeFactors;
        }
        
        double priceContract() {
            int n = basket.size();
            if (n == 0) throw std::runtime_error("No ValueNotes");
            if (relativeFactors.size() != size_t(n)) calculateRelativeFactors();
            
            // Quadratic approximation coefficients for each asset
            struct Coeff { double a, b, c; };
            std::vector<Coeff> coeffs(n);
            
            // Numerical integration setup
            const int NPOINTS = 2000;
            double zmin = -3, zmax = 3, dz = (zmax - zmin) / (NPOINTS - 1);
            
            // Calculate quadratic approximations Q_i(z) = a*z^2 + b*z + c
            for (int i = 0; i < n; ++i) {
                long double Sw = 0, Swz = 0, Swz2 = 0, Swz3 = 0, Swz4 = 0;
                long double SwQ = 0, SwzQ = 0, Swz2Q = 0;
                
                for (int k = 0; k < NPOINTS; ++k) {
                    double z = zmin + k * dz;
                    long double phi = 1 / std::sqrt(2 * M_PI) * std::expl(-0.5L * z * z);
                    
                    double E0 = basket[i]->currentEffectiveRate;
                    double sigma = basket[i]->volatility;
                    long double ERt = E0 * std::expl((sigma / 100.0L) * std::sqrt(expiration) * z
                                                   - 0.5L * (sigma / 100.0L) * (sigma / 100.0L) * expiration);
                    long double Pt = basket[i]->priceGivenYield((double)ERt, expiration);
                    long double Q = relativeFactors[i] > 1e-12 ? Pt / relativeFactors[i] : 0;
                    
                    // Accumulate moments for least squares fitting
                    Sw += phi; Swz += phi * z; Swz2 += phi * z * z;
                    Swz3 += phi * z * z * z; Swz4 += phi * z * z * z * z;
                    SwQ += phi * Q; SwzQ += phi * z * Q; Swz2Q += phi * z * z * Q;
                }
                
                // Solve 3x3 system for quadratic coefficients
                long double A11 = Swz4, A12 = Swz3, A13 = Swz2;
                long double A21 = Swz3, A22 = Swz2, A23 = Swz;
                long double A31 = Swz2, A32 = Swz, A33 = Sw;
                long double B1 = Swz2Q, B2 = SwzQ, B3 = SwQ;
                
                // Convert to double precision for Gaussian elimination
                double a11 = (double)A11, a12 = (double)A12, a13 = (double)A13;
                double a21 = (double)A21, a22 = (double)A22, a23 = (double)A23;
                double a31 = (double)A31, a32 = (double)A32, a33 = (double)A33;
                double b1 = (double)B1, b2 = (double)B2, b3 = (double)B3;
                
                // Partial pivoting for numerical stability
                if (std::fabs(a11) < 1e-15) {
                    if (std::fabs(a21) > std::fabs(a11)) {
                        std::swap(a11, a21); std::swap(a12, a22); std::swap(a13, a23); std::swap(b1, b2);
                    } else if (std::fabs(a31) > std::fabs(a11)) {
                        std::swap(a11, a31); std::swap(a12, a32); std::swap(a13, a33); std::swap(b1, b3);
                    }
                }
                
                // Gaussian elimination
                double m21 = a21 / a11; a22 -= m21 * a12; a23 -= m21 * a13; b2 -= m21 * b1;
                double m31 = a31 / a11; a32 -= m31 * a12; a33 -= m31 * a13; b3 -= m31 * b1;
                
                if (std::fabs(a22) < 1e-15) {
                    std::swap(a22, a32); std::swap(a23, a33); std::swap(b2, b3);
                }
                
                double m32 = a32 / a22; a33 -= m32 * a23; b3 -= m32 * b2;
                
                // Back substitution
                double c = std::fabs(a33) < 1e-15 ? 0 : b3 / a33;
                double bcoef = std::fabs(a22) < 1e-15 ? 0 : (b2 - a23 * c) / a22;
                double acoef = std::fabs(a11) < 1e-15 ? 0 : (b1 - a12 * bcoef - a13 * c) / a11;
                
                coeffs[i] = {acoef, bcoef, c};
            }
            
            // Find intersection points of quadratic approximations
            std::vector<double> zPts;
            zPts.push_back(-3); zPts.push_back(3);
            
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    double A = coeffs[i].a - coeffs[j].a;
                    double B = coeffs[i].b - coeffs[j].b;
                    double C = coeffs[i].c - coeffs[j].c;
                    
                    if (std::fabs(A) < 1e-12) {
                        // Linear case
                        if (std::fabs(B) < 1e-12) continue;
                        double zint = -C / B;
                        if (zint >= -3 - 1e-9 && zint <= 3 + 1e-9) {
                            zPts.push_back(zint);
                        }
                    } else {
                        // Quadratic case
                        double disc = B * B - 4 * A * C;
                        if (disc < 0) continue;
                        double sd = std::sqrt(disc);
                        double z1 = (-B + sd) / (2 * A);
                        double z2 = (-B - sd) / (2 * A);
                        if (z1 >= -3 - 1e-9 && z1 <= 3 + 1e-9) zPts.push_back(z1);
                        if (z2 >= -3 - 1e-9 && z2 <= 3 + 1e-9) zPts.push_back(z2);
                    }
                }
            }
            
            // Sort and remove duplicates
            std::sort(zPts.begin(), zPts.end());
            zPts.erase(std::unique(zPts.begin(), zPts.end(),
                                 [](double a, double b) { return std::fabs(a - b) < 1e-6; }),
                      zPts.end());
            
            if (zPts.front() > -3 + 1e-6) zPts.insert(zPts.begin(), -3);
            if (zPts.back() < 3 - 1e-6) zPts.push_back(3);
            
            // Initialize result vectors
            deliveryProb.assign(n, 0.0);
            sensitivityVol.assign(n, 0.0);
            sensitivityPrice.assign(n, 0.0);
            
            // Calculate derivatives for sensitivity analysis
            std::vector<double> dER0dVP0(n);
            for (int i = 0; i < n; ++i) {
                double dPdE = basket[i]->derivativePriceToYield(basket[i]->currentEffectiveRate, 0.0);
                dER0dVP0[i] = std::fabs(dPdE) < 1e-12 ? 0 : 1.0 / dPdE;
            }
            
            long double contractPV = 0;
            
            // Process each interval between intersection points
            for (size_t idx = 0; idx < zPts.size() - 1; ++idx) {
                double zL = zPts[idx], zR = zPts[idx + 1];
                if (zR <= zL) continue;
                
                // Find cheapest asset in this interval
                double zM = 0.5 * (zL + zR);
                double minR = 1e300;
                int ci = 0;
                for (int i = 0; i < n; ++i) {
                    double Q = coeffs[i].a * zM * zM + coeffs[i].b * zM + coeffs[i].c;
                    if (Q < minR) {
                        minR = Q;
                        ci = i;
                    }
                }
                
                // Standard normal PDF and CDF
                auto phi = [](double z) { return 1.0 / sqrt(2 * M_PI) * exp(-0.5 * z * z); };
                auto Phi = [](double z) { return 0.5 * (1 + erf(z / sqrt(2.0))); };
                
                // Analytical integration for contract value
                double a = coeffs[ci].a, b = coeffs[ci].b, c = coeffs[ci].c, K = 100.0;
                double phiL = phi(zL), phiR = phi(zR), PhiL = Phi(zL), PhiR = Phi(zR);
                
                long double t1 = a * (zL * phiL - zR * phiR + (PhiR - PhiL));
                long double t2 = b * (phiL - phiR);
                long double t3 = (c - K) * (PhiR - PhiL);
                
                contractPV += relativeFactors[ci] * (t1 + t2 + t3);
                deliveryProb[ci] += (PhiR - PhiL);
                
                // Sensitivity calculations using Simpson's rule
                const int SUB = 100;
                double hz = (zR - zL) / SUB;
                long double segS = 0, segE = 0;
                
                for (int k = 0; k <= SUB; ++k) {
                    double z = zL + k * hz;
                    long double w = (k == 0 || k == SUB) ? 1 : (k % 2 == 0 ? 2 : 4);
                    long double pv = 1.0L / sqrt(2 * M_PI) * std::expl(-0.5L * z * z);
                    
                    double E0 = basket[ci]->currentEffectiveRate;
                    double sigma = basket[ci]->volatility;
                    long double ERt = E0 * std::expl((sigma / 100.0L) * std::sqrt(expiration) * z
                                                   - 0.5L * (sigma / 100.0L) * (sigma / 100.0L) * expiration);
                    long double Pt = basket[ci]->priceGivenYield((double)ERt, expiration);
                    long double dPdE = basket[ci]->derivativePriceToYield((double)ERt, expiration);
                    
                    // Sensitivity to volatility
                    long double dERdS = ERt * (std::sqrt(expiration) * z - (sigma / 100.0L) * expiration);
                    long double dPdS = dPdE * dERdS;
                    
                    // Sensitivity to initial effective rate
                    long double dERdE0 = E0 != 0 ? ERt / E0 : 0;
                    long double dPdE0 = dPdE * dERdE0;
                    
                    segS += w * dPdS * pv;
                    segE += w * dPdE0 * pv;
                }
                
                segS *= hz / 3;
                segE *= hz / 3;
                sensitivityVol[ci] += (double)segS;
                sensitivityPrice[ci] += (double)segE;
            }
            
            // Apply discount factor
            long double dfact = std::expl(-riskFreeRate / 100.0 * expiration);
            contractPV *= dfact;
            
            for (int i = 0; i < n; ++i) {
                sensitivityVol[i] *= (double)dfact;
                sensitivityPrice[i] *= (double)dfact * dER0dVP0[i];
            }
            
            // Normalize probabilities
            long double tot = 0;
            for (auto& p : deliveryProb) tot += p;
            if (tot > 1e-6) for (auto& p : deliveryProb) p /= tot;
            
            return (double)contractPV;
        }
    };

    /**
     * Demonstration function for Task 2
     * Shows delivery contract pricing and sensitivity analysis
     */
    void run() {
        std::cout << "=== Task 2: DeliveryContract Demo ===\n";
        
        ValueNoteCumulative vn1(100, 5.0, 3.5, 1, 1.5);
        ValueNoteCumulative vn2(100, 1.5, 2.0, 2, 2.5);
        ValueNoteCumulative vn3(100, 4.5, 3.25, 1, 1.5);
        ValueNoteCumulative vn4(100, 10.0, 8.0, 4, 5.0);

        DeliveryContract contract(5.0, 0.25, 4.0);
        contract.addValueNote(&vn1);
        contract.addValueNote(&vn2);
        contract.addValueNote(&vn3);
        contract.addValueNote(&vn4);

        contract.calculateRelativeFactors();
        auto RFs = contract.getRelativeFactors();
        
        std::cout << std::fixed << std::setprecision(4) << "Relative Factors: ";
        for (size_t i = 0; i < RFs.size(); ++i) {
            std::cout << "VN" << (i + 1) << "=" << RFs[i]
                      << (i + 1 < RFs.size() ? ", " : "");
        }
        std::cout << "\n";

        double price = contract.priceContract();
        std::cout << "Contract Price = " << price << "\n";

        std::cout << "Delivery Probabilities: ";
        for (size_t i = 0; i < contract.deliveryProb.size(); ++i) {
            std::cout << "VN" << (i + 1) << "=" << contract.deliveryProb[i]
                      << (i + 1 < contract.deliveryProb.size() ? ", " : "");
        }
        std::cout << "\n";

        std::cout << "Sens. to vol (∂P/∂σ): ";
        for (size_t i = 0; i < contract.sensitivityVol.size(); ++i) {
            std::cout << "VN" << (i + 1) << "=" << contract.sensitivityVol[i]
                      << (i + 1 < contract.sensitivityVol.size() ? ", " : "");
        }
        std::cout << "\n";

        std::cout << "Sens. to init price (∂P/∂VP₀): ";
        for (size_t i = 0; i < contract.sensitivityPrice.size(); ++i) {
            std::cout << "VN" << (i + 1) << "=" << contract.sensitivityPrice[i]
                      << (i + 1 < contract.sensitivityPrice.size() ? ", " : "");
        }
        std::cout << "\n\n";
    }

} // namespace Task2

/**
 * Main function demonstrating both tasks
 */
int main() {
    Task1::run();
    Task2::run();
    return 0;
}
