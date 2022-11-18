import unicodedata


class DameUtils():

    def dicc_dataset(self, sex):
        # given the gender (sex) returns a dictionary where the key is
        # the country and the value is the path with names and sex
        if ((sex == "male") or (sex == "males") or (sex == 1)):
            path = {"ar": "files/names/names_ar/armales.csv",
                    "at": "files/names/names_at/atmales.csv",
                    "au": "files/names/names_au/aumales.csv",
                    "be": "files/names/names_be/bemales.csv",
                    "ca": "files/names/names_ca/camales.csv",
                    "ch": "files/names/names_ch/chmales.csv",
                    "cn": "files/names/names_cn/cnmales.csv",
                    "de": "files/names/names_de/demales.csv",
                    "dk": "files/names/names_dk/dkmales.csv",
                    "es": "files/names/names_es/esmales.csv",
                    "fi": "files/names/names_fi/fimales.csv",
                    "fr": "files/names/names_fr/frmales.csv",
                    "gb": "files/names/names_gb/gbmales.csv",
                    "ie": "files/names/names_ie/iemales.csv",
                    "ine": "files/names/names_es/esmales.csv",
                    "inter": "files/names/names_inter/intermales.csv",
                    "is": "files/names/names_is/ismales.csv",
                    "it": "files/names/names_it/itmales.csv",
                    "mx": "files/names/names_mx/mxmales.csv",
                    "no": "files/names/names_no/nomales.csv",
                    "nz": "files/names/names_nz/nzmales.csv",
                    "pt": "files/names/names_pt/ptmales.csv",
                    "ru": "files/names/names_ru/rumales.csv",
                    "ru_ru": "files/names/names_ru/rumales.csv",
                    "ru_en": "files/names/names_ru/rumales.en.csv",
                    "se": "files/names/names_se/semales.csv",
                    "si": "files/names/names_si/simales.csv",
                    # "tr": "files/names/names_tr/trmales.csv",
                    "us": "files/names/names_us/usmales.csv",
                    "usa": "files/names/names_us/usmales.csv",
                    "uy": "files/names/names_uy/uymales.csv"}
        elif ((sex == "female") or (sex == "females") or (sex == 0)):
            path = {"ar": "files/names/names_ar/arfemales.csv",
                    "at": "files/names/names_at/atfemales.csv",
                    "au": "files/names/names_au/aufemales.csv",
                    "be": "files/names/names_be/befemales.csv",
                    "ca": "files/names/names_ca/cafemales.csv",
                    "ch": "files/names/names_ch/chfemales.csv",
                    "cn": "files/names/names_cn/cnfemales.csv",
                    "de": "files/names/names_de/defemales.csv",
                    "dk": "files/names/names_dk/dkfemales.csv",
                    "es": "files/names/names_es/esfemales.csv",
                    "fi": "files/names/names_fi/fifemales.csv",
                    "fr": "files/names/names_fr/frfemales.csv",
                    "gb": "files/names/names_gb/gbfemales.csv",
                    "ie": "files/names/names_ie/iefemales.csv",
                    "ine": "files/names/names_es/esfemeninos.csv",
                    "inter": "files/names/names_inter/interfemales.csv",
                    "is": "files/names/names_is/isfemales.csv",
                    "it": "files/names/names_it/itfemales.csv",
                    "mx": "files/names/names_mx/mxfemales.csv",
                    "no": "files/names/names_no/nofemales.csv",
                    "nz": "files/names/names_nz/nzfemales.csv",
                    "pt": "files/names/names_pt/ptfemales.csv",
                    "ru": "files/names/names_ru/rufemales.csv",
                    "ru_ru": "files/names/names_ru/rufemales.csv",
                    "ru_en": "files/names/names_ru/rufemales.en.csv",
                    "se": "files/names/names_se/sefemales.csv",
                    "si": "files/names/names_si/sifemales.csv",
                    # "tr": "files/names/names_tr/trfemales.csv",
                    "us": "files/names/names_us/usfemales.csv",
                    "usa": "files/names/names_us/usfemales.csv",
                    "uy": "files/names/names_uy/uyfemales.csv"}
        elif ((sex == "all") or (sex == 2)):
            path = {"ar": "files/names/names_ar/arall.csv",
                    "at": "files/names/names_at/atall.csv",
                    "au": "files/names/names_au/auall.csv",
                    "be": "files/names/names_be/beall.csv",
                    "ca": "files/names/names_ca/caall.csv",
                    "cn": "files/names/names_cn/cnall.csv",
                    "de": "files/names/names_de/deall.csv",
                    "dk": "files/names/names_dk/dkall.csv",
                    "es": "files/names/names_es/esall.csv",
                    "fi": "files/names/names_fi/fiall.csv",
                    "fr": "files/names/names_fr/frall.csv",
                    "gb": "files/names/names_gb/gball.csv",
                    "ie": "files/names/names_ie/ieall.csv",
                    "ine": "files/names/names_es/esall.csv",
                    "inter": "files/names/names_inter/interall.csv",
                    "is": "files/names/names_is/isall.csv",
                    "it": "files/names/names_it/itall.csv",
                    "mx": "files/names/names_mx/mxall.csv",
                    "no": "files/names/names_no/noall.csv",
                    "nz": "files/names/names_nz/nzall.csv",
                    "pt": "files/names/names_pt/ptall.csv",
                    "ru": "files/names/names_ru/ruall.csv",
                    "ru_ru": "files/names/names_ru/ruall.csv",
                    "ru_en": "files/names/names_ru/ruall.en.csv",
                    "se": "files/names/names_si/seall.csv",
                    "si": "files/names/names_si/siall.csv",
                    # "tr": "files/names/names_tr/trall.csv",
                    "us": "files/names/names_us/usall.csv",
                    "usa": "files/names/names_us/usall.csv",
                    "uy": "files/names/names_uy/uyall.csv"}
        return path

    def drop_dots(self, s):
        # given s removes dots symbols in the string
        aux = ""
        for c in unicodedata.normalize('NFD', str(s)):
            if (c != '.'):
                aux = aux + c
        return aux

    def force_whitespaces(self, s):
        # replace underscore, hyphens, ... by white spaces
        aux = ""
        for c in unicodedata.normalize('NFD', str(s)):
            if ((c == "_") or (c == '-')):
                aux = aux + " "
            else:
                aux = aux + c
        return aux

    def drop_accents(self, s):
        # given a string s delete accents
        return ''.join((c for c in unicodedata.normalize('NFD', s)
                        if unicodedata.category(c) != 'Mn'))
