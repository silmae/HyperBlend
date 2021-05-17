
class RenderParametersForSeries:

    clear_on_start = True
    clear_references = True
    render_references = True
    dry_run = False
    wl_list = None
    abs_dens_list = None
    scat_dens_list = None
    scat_ai_list = None
    mix_fac_list = None


    def get_as_dict(self):
        d = {
            'clear_on_start': self.clear_on_start,
            'clear_references': self.clear_references,
            'render_references': self.render_references,
            'dry_run': self.dry_run,
            'wl_list': self.wl_list,
            'abs_dens_list': self.abs_dens_list,
            'scat_dens_list': self.scat_dens_list,
            'scat_ai_list': self.scat_ai_list,
            'mix_fac_list': self.mix_fac_list,
        }
        return d

    def get_single(self, i):
        if i >= len(self.wl_list):
            raise IndexError(f"Index {i} out of bounds for list of {len(self.wl_list)}.")
        else:
            rps = RenderParametersForSingle()
            # rps.clear_rend_folder = self.clear_on_start
            rps.clear_references = self.clear_references
            rps.render_references = self.render_references
            rps.dry_run = self.dry_run
            rps.wl = self.wl_list[i]
            rps.abs_dens = self.abs_dens_list[i]
            rps.scat_dens = self.scat_dens_list[i]
            rps.scat_ai = self.scat_ai_list[i]
            rps.mix_fac = self.mix_fac_list[i]
            return rps



class RenderParametersForSingle:

    clear_rend_folder = False
    clear_references = False
    render_references = True
    dry_run = False

    wl = None
    abs_dens = None
    scat_dens = None
    scat_ai = None
    mix_fac = None