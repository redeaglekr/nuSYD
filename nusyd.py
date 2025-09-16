# Import the usual suspects ...
import sys
sys.path.append("/home/redaegle/gen/lib64/python3.12/site-packages/")
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft, Gaussian1DKernel
from astropy.timeseries import LombScargle
import time as tm
from joblib import Parallel, delayed

class nuSYD:
    def __init__(self,  time, flux,  guess_numax=30, name = "Star", lc_type="Kepler", 
                 factor=2, no_it=5, mc_iter = False,  apodized=False, plot=False):
        """
        Parameters
        ----------
        t : array
            Time series (days).
        m : array
            Normalized flux (relative).
        lc_type : str
            Type of light curve ("Kepler", "TESS", etc.).
        guess_numax : float
            Initial guess for numax (µHz).
        factor : int
            Kernel width scaling factor.
        no_it : int
            Number of iterations in refinement loop.
        apodized : bool
            Apply sinc correction near Nyquist if True.
        plot : bool
            Plot diagnostic figures if True.
        """
        self.time = time
        self.flux = flux
        self.lc_type = lc_type
        self.guess_numax = guess_numax
        self.factor = factor
        self.no_it = no_it
        self.apodized = apodized
        self.plot = plot
        self.mc_iter = mc_iter
        if self.lc_type == "Kepler":
            self.name = f"KIC {name}"
        elif self.lc_type == "TESS":
            self.name = f"TIC {name}"
            
        
        # results
        self.freq = None
        self.power = None
        self.numax = None
        self.widths = None
        self.history = None
        self.errors = None

    # ------------------------------------------------------
    # 1. Lomb-Scargle PDS
    # ------------------------------------------------------
    def _calc_lomb_scargle(self, t, y):
        oversample = 10
        df = 1.0 / (t.max() - t.min())
        fmin, fmax = df, 24
        freq = np.arange(fmin, fmax, df / oversample)

        model = LombScargle(t, y)
        sc = model.power(freq, method="fast", normalization="psd")
        fct = np.sqrt(4. / len(t))
        amp = np.sqrt(sc) * fct

        return freq * 11.574, amp * 1e6  # µHz, ppm
    
    # ------------------------------------------------------
    # 2. Width calculation at half-maximum
    # ------------------------------------------------------
    def _width_calc(self, frequency, smoothed_power, numax):
        idx = np.where((frequency >= 0.1 * numax) & (frequency <= 1.9 * numax))
        freqs, smtd_pwr = frequency[idx], smoothed_power[idx]
        ps_max = max(smtd_pwr)
        subtracted_power = np.sign(smtd_pwr - (0.5 * ps_max))
        sc = ((np.roll(subtracted_power, 1) - subtracted_power) != 0).astype(int)
        inds = np.where(sc == 1)
        inds_sign = freqs[inds]

        neighbours = sorted(np.insert(inds_sign, -1, numax))
        i_numax = np.where(neighbours == numax)[0][0]

        if neighbours[-1] == numax:
            widl = neighbours[i_numax - 1]
            widr = numax + (numax - widl)
        else:
            widl = neighbours[i_numax - 1]
            widr = neighbours[i_numax + 1]

        return widl, widr

    # ------------------------------------------------------
    # 3. Effective observing time
    # ------------------------------------------------------
    def _eff(self, time):
        reg_bounds = [2035.5, 2824.563]
        regions = [
            time[time <= reg_bounds[0]],
            time[(time >= reg_bounds[0]) & (time <= reg_bounds[1])],
            time[time>= reg_bounds[1]]
        ]
        eff_time = 0
        for r in regions:
            if len(r) > 1:
                eff_time += len(r) * np.median(np.diff(r))
        return eff_time
    # ------------------------------------------------------
    # 4. Initial numax (if required by user)
    # ------------------------------------------------------
    def _get_numax_frompds(self, freq, power, lc_type, nyq):
        if self.lc_type == "TESS":

            inds1 = np.where((freq >= 5) & (freq <= nyq))
            pds1 = power[inds1]
            pow1 = np.nanmean(pds1)


            inds2 = np.where((freq >=0.91*nyq) & (freq <=nyq))
            pds2 = power[inds2]
            pow2 = np.nanmean(pds2)
            sid = np.nanstd(pds2)

            dp= abs(pow1-pow2)
            lognumax = (-1.039*np.log10(dp)) + 5.1594
            guess_numax = 10**lognumax
        elif self.lc_type == "Kepler":
            inds1 = np.where((freq >= 5) & (freq <= nyq))
            pds1 = power[inds1]
            pow1 = np.nanmean(pds1)


            inds2 = np.where((freq >=0.97*nyq) & (freq <=nyq))
            pds2 = power[inds2]
            pow2 = np.nanmean(pds2)
            sid = np.nanstd(pds2)

            dp= abs(pow1-pow2)
            lognumax = (-0.115*(np.log10(dp))**2) + (-0.0824*np.log10(dp))+3.08

            guess_numax = 10**lognumax
        
        return guess_numax
            
            
        
    # ------------------------------------------------------
    # 5. numax refinement loop
    # ------------------------------------------------------
    
    def _refine_numax(self, freq, power, guess_numax, nyq):
        
        fres = freq[1] - freq[0]
        sinc = np.sin(np.pi * freq / (2 * nyq)) / (np.pi * freq / (2 * nyq))
            
        numax = guess_numax
        numaxs, powers, widths  = [], [], []

        for it in range(self.no_it):
            # White noise region
            if self.lc_type == "Kepler":
                
                if numax < 0.9 * nyq:
                    wn_l, wn_h = 0.97 * nyq, nyq
                else:
                    wn_l, wn_h = 0.68 * numax, 0.72 * numax

            elif self.lc_type == "TESS":
                if numax < 0.8 * nyq:
                    wn_l, wn_h = 0.91 * nyq, nyq
                else:
                    wn_l, wn_h = 0.68 * numax, 0.72 * numax
                
            inds = np.where((freq >= wn_l) & (freq <= wn_h))
            white_noise = np.nanmean(power[inds])

            # Background model
            background = (freq / numax) ** -2
            background[freq >= nyq] = 1e9

            # Power prep
            sub_power = power - white_noise
            if self.apodized and it == (self.no_it - 1):
                sub_power /= sinc ** 2
            divided_power = sub_power / background

            # Smooth
            kernel_width = self.factor * 0.26 * numax ** 0.772
            gausk = Gaussian1DKernel(kernel_width / (2.355 * fres))
            smoothed_power = convolve_fft(divided_power, gausk)

            # Restrict around numax
            fmask = np.where((freq <= nyq) &
                             (freq >= 0.6 * numax) &
                             (freq <= min(1.4 * numax, nyq)))
            freq_sub, smt_sub = freq[fmask], smoothed_power[fmask]

            # Peak location
            cc = np.argmax(smt_sub)
            numax = freq_sub[cc]
            ppeak = smt_sub[cc]

            # Width
            try:
                widl, widr = self._width_calc(freq, smoothed_power, numax)
            except Exception:
                widl, widr = np.nan, np.nan

            numaxs.append(numax)
            powers.append(ppeak)
            widths.append(widr-widl)
        

        
        numax_uncor_final = numax

        width_final = widr - widl

        ######Bias correction###############
        
        
        sigma_actual = 0.248*((width_final)**1.08)  #(widr1-widl1)/2.355#

        numax_actual =  (numax_uncor_final**2 - 2*(sigma_actual)**2)/numax_uncor_final

        return numax_actual, numax_uncor_final, ppeak, widl, widr, smoothed_power, divided_power

    # ------------------------------------------------------
    # 6. Main pipeline runner
    # ------------------------------------------------------
    def run(self):
        print(f"<-----Started the analysis for {self.name} ---->")
        nyq = 24 * 11.574 if self.lc_type in ["Kepler", "TESS"] else None
        time_eff = self._eff(self.time) if self.lc_type == "TESS" else len(self.time)*np.median(np.diff(self.time))

        # PDS
        freq, amp = self._calc_lomb_scargle(self.time, self.flux)
        power = (amp ** 2) * (time_eff * 24 * 3600) / 1e6

        # Initial numax
        if self.guess_numax == "from_lc":
            guess_numax =  self._get_numax_frompds(freq, power,self.lc_type,   nyq)
        self.guess_numax = guess_numax

        # core nuSYD
        numax, numax_uncor, ppeak, width_l, width_r, smoothed_power, divided_power = self._refine_numax(freq, power, self.guess_numax, nyq)
        if np.isnan(width_r - width_l) == False:
            print("Width calculation is succesful!! Bias correction can be trusted.")

        # Save results
        self.freq = freq
        self.power = power
        self.numax_uncor = numax_uncor
        self.numax = numax
        self.widths = (width_l, width_r)
        self.smoothed_power = smoothed_power
        self.divided_power = divided_power
        self.ppeak = ppeak

        if self.mc_iter:
            print("<========= Running MC sampling ===========>")
            #time1 = tm.time()
            self.errors = self.montecarlo(freq, power,self.numax, nyq)
            #time2 = tm.time()
            #print("time", time2-time1)
        else:
            self.errors = np.nan
            

        if self.plot:
            self._plot_results()

        return {
            "freq": self.freq,
            "power": self.power,
            "numax": self.numax,
            "widths": self.widths,
            "errors": self.errors
        }

    # ------------------------------------------------------
    # 7. Plotting
    # ------------------------------------------------------
    def _plot_results(self):
        fig, axs = plt.subplots(2, 1, figsize = (9, 6))

        max_power = np.max(self.power[(self.freq > 0.6*self.numax_uncor) & (self.freq < 1.5*self.numax_uncor)])
        pf = max_power/self.ppeak ## plotting factor to enlarge the smoothed power


        axs[0].plot(self.freq, self.power, c="tab:blue")
        axs[0].axvline(self.guess_numax,color = "blue", ls = "dashed", label = r"Initial guess for $\nu_{max}$ = "+str(round(self.guess_numax, 4))+r"$\rm \mu$Hz")
        axs[0].legend()
                       
        axs[1].plot(self.freq, self.divided_power, c="tab:blue")
        axs[1].plot(self.freq, pf*self.smoothed_power, c="tab:orange")
        axs[1].scatter(self.numax_uncor, pf*self.ppeak, c = "tab:red", zorder =2)
        axs[1].axvline(self.numax_uncor, color="tab:red", linestyle="--", label=r"$\nu_{\mathrm{max}}$ (uncorrected) = "+str(round(self.numax_uncor, 4))+r" $\rm \mu$Hz")
        axs[1].axvline(self.numax, color="black", linestyle="--", label=r"$\nu_{\mathrm{max}}$ (final) = "+str(round(self.numax, 4))+r" $\rm \mu$ Hz")
        axs[1].scatter((self.widths[0], self.widths[1]), (pf*self.ppeak/2, pf*self.ppeak/2), color = "tab:green")
        axs[1].plot((self.widths[0], self.widths[1]), (pf*self.ppeak/2, pf*self.ppeak/2), color = "tab:green", ls = "dotted", label = r"Width = {:.4f} $\rm \mu$Hz".format(self.widths[ 1]-self.widths[0]))
        axs[1].legend(loc = "upper right")
        axs[1].set_xlim(0.4*self.numax_uncor, 1.6*self.numax_uncor)
        axs[1].set_ylim(0, 1.1*max_power)


        fig.supxlabel(r"Frequency ($\rm \mu$Hz)")
        fig.supylabel(r"Power density (ppm$^{2}$/$\rm \mu$Hz)")
        fig.suptitle(self.name)
        if self.plot:
            plt.show()
    # ----------------------------------------------------------
    # 8. Error bars
    # ----------------------------------------------------------

    def montecarlo(self, freq, power, numax, nyq):
        fres = freq[1] - freq[0]
        kernel_width = self.factor * 0.26 * numax ** 0.772
        gausk = Gaussian1DKernel(kernel_width / (2.355 * fres))

        background = (freq / numax) ** -2
        background[freq >= nyq] = 1e9

        fmask = np.where((freq <= nyq) &
                         (freq >= 0.6 * numax) &
                         (freq <= min(1.4 * numax, nyq)))

        # Define single Monte Carlo iteration
        def _single_mc(_):
            new_power = (np.random.chisquare(2, len(freq)) * power) / 2

            if self.lc_type == "Kepler":
                wn_l, wn_h = (0.97 * nyq, nyq) if numax < 0.9 * nyq else (0.68 * numax, 0.72 * numax)
            else:  
                wn_l, wn_h = (0.91 * nyq, nyq) if numax < 0.8 * nyq else (0.68 * numax, 0.72 * numax)

            inds = np.where((freq >= wn_l) & (freq <= wn_h))
            white_noise = np.nanmean(new_power[inds])

            divided_power = (new_power - white_noise) / background

            smoothed_power = convolve_fft(divided_power, gausk)

            freq_sub, smt_sub = freq[fmask], smoothed_power[fmask]
            cc = np.argmax(smt_sub)
            numax_uncor_final = freq_sub[cc]

            try:
                widl, widr = self._width_calc(freq, smoothed_power, numax)
            except Exception:
                widl, widr = np.nan, np.nan

            width_final = widr - widl
            sigma_actual = 0.248 * (width_final**1.08)
            numax_actual = (numax_uncor_final**2 - 2*(sigma_actual)**2) / numax_uncor_final

            return numax_actual

       
        results = Parallel(n_jobs=5, backend = 'threading')(delayed(_single_mc)(i) for i in range(self.mc_iter))

        error = np.nanstd(results) / np.sqrt(self.mc_iter)

        return error
            

       
        


##name =149901871
##tic = name
##
##
##
##time = np.load("/home/redaegle/usyd/tess_cvz/qlp_ctd_5d_rgcut_251124/TIC"+str(int(tic))+"_tess_qlp_ctd_5d.npy")[:,0]  ######your lightcurve file, for me it is this one.  Other users can give their own path
##flux = np.load("/home/redaegle/usyd/tess_cvz/qlp_ctd_5d_rgcut_251124/TIC"+str(int(tic))+"_tess_qlp_ctd_5d.npy")[:,1]  ######your lightcurve file
##        
##
##pipeline = nuSYD(time, flux, name = name,  lc_type="TESS",
##                              guess_numax="from_lc", mc_iter=200,  plot=True)
##result = pipeline.run()

