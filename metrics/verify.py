import torch


class verify_(object):
    """根据预测及对应的观测计算各种评估指标

        +-------------+-------------------------------+
        |             |           Observed            |
        |             +---------------+---------------+
        |             |      Y        |      N        |
        +---------+---+---------------+---------------+
        |         | Y |      hits     |  false alarms |
        |Predicted+---+---------------+---------------+
        |         | N |     misses    | correct neg   |
        +---------+---+---------------+---------------+

        hits         -> ftot
        false alarms -> ftof
        misses       -> ffot
        correct neg  -> ffof

        Reference
          - https://www.cawcr.gov.au/projects/verification/
    """

    def __init__(self, obs=None, pred=None, thre_min=None, thre_max=None,
                 hits=None, false=None, misses=None, corrneg=None, mark_key=""):
        """
        :param obs(torch.tensor): 观测
        :param pred(torch.tensor): 预测
        :param thre_min(float): 最小阈值
        :param thre_max(float): 最大阈值
        :param hits(int): 命中数
        :param false(int): 虚警数
        :param misses(int): 漏报数
        :param corrneg(int): correct negative

        如果提供了观测和预测值，则不需要提供命中数等信息，程序会根据提供的阈值自动计算
        """
        if obs is not None and pred is not None:
            ftot, ftof, ffot, ffof = self.get_hit_counts(obs, pred,
                                                         thre_min=thre_min,
                                                         thre_max=thre_max)

            self.ftot = ftot
            self.ftof = ftof
            self.ffot = ffot
            self.ffof = ffof
            self.rmse = (((obs - pred)**2).mean())**0.5
            self.mae = (torch.abs(obs - pred)).mean()

        elif obs is None and pred is None:
            if hits is None or false is None or misses is None or corrneg is None:
                raise ValueError(
                    'hits, false alarms, misses and correct negative must be provided when obs and pred is None.')
            else:
                self.ftot = hits
                self.ftof = false
                self.ffot = misses
                self.ffof = corrneg
        else:
            raise ValueError('obs and pred should not be None.')

        self.mark_key = mark_key  # 标记dicts

    def get_hit_counts(self, obs=None, pred=None, thre_min=None, thre_max=None):
        """
        """
        if thre_min is None and thre_max is None:
            raise ValueError(f'thre_min and thre_max should be not None at least!')
        elif thre_min is not None and thre_max is None:
            pred = pred >= thre_min
            obse = obs >= thre_min
        elif thre_min is not None and thre_max is not None:
            pred = (pred >= thre_min) & (pred < thre_max)
            obse = (obs >= thre_min) & (obs < thre_max)
        else:
            raise ValueError(f'the min and max of threshold is {thre_min} and {thre_max}, respectively.')

        fToT = torch.logical_and(pred, torch.logical_and(pred, obse)).sum().float()
        fToF = torch.logical_and(pred, torch.logical_and(pred, ~obse)).sum().float()
        fFoT = torch.logical_and(~pred, torch.logical_and(~pred, obse)).sum().float()
        fFoF = torch.logical_and(~pred, torch.logical_and(~pred, ~obse)).sum().float()

        return fToT, fToF, fFoT, fFoF

    def describe(self, ftot=None, ffof=None, ftof=None, ffot=None):
        '''输出各种指标的信息，包括准确率、HSS、CSI、POD、FAR、SR等

        :param ftot, ffof, ftof, ffot 表示命中数等，可单独给出或通过观测和预测信息计算
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        if ffof is None:
            ffof = self.ffof

        results = {'Accuracy_%s' % self.mark_key: self.accu(ftot, ffof, ftof, ffot),
                   'HSS_%s' % self.mark_key: self.HSS(ftot, ftof, ffot, ffof),
                   'CSI_%s' % self.mark_key: self.CSI(ftot, ftof, ffot),
                   'POD_%s' % self.mark_key: self.POD(ftot, ffot),
                   'FAR_%s' % self.mark_key: self.FAR(ftof, ftot),
                   'BIAS_%s' % self.mark_key: self.BIAS(ftof, ftot, ffot),
                   # 'SR_%s' % self.mark_key: self.SR(ftot, ftof),
                   # 'RMSE_%s' % self.mark_key: self.rmse,
                   # 'MAE_%s' % self.mark_key: self.mae,
                   }
        # results = {'HSS_%s' % self.mark_key: self.HSS(ftot, ftof, ffot, ffof),}

        return results

    def accu(self, ftot=None, ffof=None, ftof=None, ffot=None):
        '''Accuracy (fraction correct)

        Accuracy = (hits + correct_neg)/(total)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        if ffof is None:
            ffof = self.ffof

        return (ftot + ffof) / (ftot + ftof + ffot + ffof + 1e-5)

    def POD(self, ftot=None, ffot=None):
        '''Probability of detection

        POD = hits/(hits + misses) = ftot / (ftot + ffot)
        '''
        if ftot is None:
            ftot = self.ftot
        if ffot is None:
            ffot = self.ffot

        return ftot / (ftot + ffot + 1e-5)

    def FAR(self, ftof=None, ftot=None):
        '''False alarm ratio

        FAR = false alarms / (hits + false alarms)
            = ftof / (ftot + ftof)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof

        return ftof / (ftot + ftof + 1e-5)


    def BIAS(self, ftof=None, ftot=None, ffot=None):
        '''Bias score

        BIAS =  (hits + false alarms) / (hits + misses)
            = (ftot + ftof) / (ftot + ftof)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        return (ftot + ftof) / (ftot + ffot + 0.0001)

    def POFD(self, ftof=None, ffof=None):
        '''Probability of false detection

        POFD = false alarms / (correct neg + false alarms)
             = ftof / (ffof + ftof)
        '''
        if ftof is None:
            ftof = self.ftof
        if ffof is None:
            ffof = self.ffof

        return ftof / (ftof + ffof + 1e-5)

    def SR(self, ftot=None, ftof=None):
        '''Success ratio

        SR = hits / (hits + false alarms)
           = ftot / (ftot + ftof)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof

        return ftot / (ftot + ftof + 1e-5)

    def TS(self, ftot=None, ftof=None, ffot=None):
        '''Threat score, critical success index

        TS = hits / (hits + misses + false alarms)
           = ftot / (ftot + ffot + ftof)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot

        return ftot / (ftot + ftof + ffot + 1e-5)

    def CSI(self, ftot=None, ftof=None, ffot=None):
        '''critical success index

        CSI = hits / (hits + misses + false alarms)
            = ftot / (ftot + ffot + ftof)
        '''
        return self.TS(ftot, ftof, ffot)

    def hits_random(self, ftot=None, ftof=None, ffot=None, ffof=None):
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        if ffof is None:
            ffof = self.ffof

        return ((ftot + ffot) * (ftot + ftof)) / (ftot + ftof + ffot + ffof + 1e-5)

    def HSS(self, ftot=None, ftof=None, ffot=None, ffof=None):
        '''

        HSS = ((hits + correct neg) - (expeted correct)) / (N - (expect correct))

        expect_corr = ((hits + misses)(hits + false alarms) + (correct neg + misses)(correct neg + false alarms))/N
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        if ffof is None:
            ffof = self.ffof

        n = ftot + ffot + ftof + ffof
        expect_corr = ((ftot + ffot) * (ftot + ftof) + (ffof + ftof) * (ffof + ffot)) / n

        return ((ftot + ffof) - expect_corr) / (n - expect_corr + 1e-5)

    def OR(self, pod=None, pofd=None):
        '''Odds ratio

        OR = (POD/(1-POD))/(POFD/(1-POFD))
        '''
        if pod is None:
            pod = self.pod()

        if pofd is None:
            pofd = self.pofd()

        return (pod / (1 - pod)) / (pofd / (1 - pofd) + 1e-5)

    def GSS(self, ftot=None, ftof=None, ffot=None, ffof=None):
        '''Equitable threat score, Gilbert skill score

        ETS = GSS = (hits - hits_r)/(hits + misses + false alarms - hits_r)

        hits_r = (hits + misses)(hits + false alarms)/(total)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        if ffof is None:
            ffof = self.ffof

        hits_r = self.hits_random(ftot, ftof, ffot, ffof)

        return (ftot - hits_r) / (ftot + ftof + ffot - hits_r + 1e-5)

class verify_time_test(object):
    """根据预测及对应的观测计算各种评估指标

        +-------------+-------------------------------+
        |             |           Observed            |
        |             +---------------+---------------+
        |             |      Y        |      N        |
        +---------+---+---------------+---------------+
        |         | Y |      hits     |  false alarms |
        |Predicted+---+---------------+---------------+
        |         | N |     misses    | correct neg   |
        +---------+---+---------------+---------------+

        hits         -> ftot
        false alarms -> ftof
        misses       -> ffot
        correct neg  -> ffof

        Reference
          - https://www.cawcr.gov.au/projects/verification/
    """
    def __init__(self, obs=None, pred=None, thre_min=None, thre_max=None,
                 hits=None, false=None, misses=None, corrneg=None, mark_key=""):
        """
        :param obs(torch.tensor): 观测
        :param pred(torch.tensor): 预测
        :param thre_min(float): 最小阈值
        :param thre_max(float): 最大阈值
        :param hits(int): 命中数
        :param false(int): 虚警数
        :param misses(int): 漏报数
        :param corrneg(int): correct negative

        如果提供了观测和预测值，则不需要提供命中数等信息，程序会根据提供的阈值自动计算
        """
        if obs is not None and pred is not None:
            ftot, ftof, ffot, ffof = self.get_hit_counts(obs, pred,
                                                         thre_min=thre_min,
                                                         thre_max=thre_max)

            self.ftot = ftot
            self.ftof = ftof
            self.ffot = ffot
            self.ffof = ffof
            self.rmse = (((obs - pred) ** 2).mean(dim=(0,2,3))) ** 0.5
            self.mae = (torch.abs(obs - pred)).mean(dim=(0,2,3))
        elif obs is None and pred is None:
            if hits is None or false is None or misses is None or corrneg is None:
                raise ValueError('hits, false alarms, misses and correct negative must be provided when obs and pred is None.')
            else:
                self.ftot = hits
                self.ftof = false
                self.ffot = misses
                self.ffof = corrneg
        else:
            raise ValueError('obs and pred should not be None.')
        self.mark_key = mark_key #标记dicts
        #print(self.describe())


    def get_hit_counts(self, obs=None, pred=None, thre_min=None, thre_max=None):
        """
        """
        if thre_min is None and thre_max is None:
            raise ValueError(f'thre_min and thre_max should be not None at least!')
        elif thre_min is not None and thre_max is None:
            pred = pred >= thre_min
            obse = obs >= thre_min
        elif thre_min is not None and thre_max is not None:
            pred = (pred >= thre_min) & (pred < thre_max)
            obse = (obs >= thre_min) & (obs < thre_max)
        else:
            raise ValueError(f'the min and max of threshold is {thre_min} and {thre_max}, respectively.')

        fToT = torch.logical_and(pred, torch.logical_and(pred, obse)).sum(dim=(0,2,3)).float()
        fToF = torch.logical_and(pred, torch.logical_and(pred, ~obse)).sum(dim=(0,2,3)).float()
        fFoT = torch.logical_and(~pred, torch.logical_and(~pred, obse)).sum(dim=(0,2,3)).float()
        fFoF = torch.logical_and(~pred, torch.logical_and(~pred, ~obse)).sum(dim=(0,2,3)).float()

        return fToT, fToF, fFoT, fFoF


    def describe(self, ftot=None, ffof=None, ftof=None, ffot=None):
        '''输出各种指标的信息，包括准确率、HSS、CSI、POD、FAR、SR等

        :param ftot, ffof, ftof, ffot 表示命中数等，可单独给出或通过观测和预测信息计算
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        if ffof is None:
            ffof = self.ffof

        HSS_results = {'HSS_time_%s_%02d'%(self.mark_key, i): score for i, score in enumerate(self.HSS(ftot, ftof, ffot, ffof))}
        CSI_results = {'TS_time_%s_%02d'%(self.mark_key, i): score for i, score in enumerate(self.TS(ftot, ftof, ffot))}
        POD_results = {'POD_time_%s_%02d' % (self.mark_key, i): score for i, score in enumerate(self.POD(ftot, ffot))}
        FAR_results = {'FAR_time_%s_%02d' % (self.mark_key, i): score for i, score in enumerate(self.FAR(ftof, ftot))}
        # SR_results = {'SR_time_%s_%02d' % (self.mark_key, i): score for i, score in enumerate(self.SR(ftot, ftof))}
        BIAS_results = {'BIAS_time_%s_%02d' % (self.mark_key, i): score for i, score in enumerate(self.BIAS(ftof, ftot, ffot))}
        ACC_results = {'ACC_time_%s_%02d' % (self.mark_key, i): score for i, score in enumerate(self.accu(ftot, ffof, ftof, ffot))}
        # RMSE_results = {'RMSE_time_%s_%02d' % (self.mark_key, i): score for i, score in
        #                enumerate(self.rmse)}
        # MAE_results = {'MAE_time_%s_%02d' % (self.mark_key, i): score for i, score in
        #                enumerate(self.mae)}

        return {**HSS_results, **CSI_results, **POD_results, **FAR_results, **BIAS_results, **ACC_results, }
        # return {**HSS_results, **CSI_results, **POD_results, **FAR_results, **SR_results, **ACC_results, **RMSE_results, **MAE_results}

    def TS(self, ftot=None, ftof=None, ffot=None):
        '''Threat score, critical success index

        TS = hits / (hits + misses + false alarms)
           = ftot / (ftot + ffot + ftof)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot

        return ftot / (ftot + ftof + ffot + 1e-5)


    def CSI(self, ftot=None, ftof=None, ffot=None):
        '''critical success index

        CSI = hits / (hits + misses + false alarms)
            = ftot / (ftot + ffot + ftof)
        '''
        return self.TS(ftot, ftof, ffot)


    def HSS(self, ftot=None, ftof=None, ffot=None, ffof=None):
        '''

        HSS = ((hits + correct neg) - (expeted correct)) / (N - (expect correct))

        expect_corr = ((hits + misses)(hits + false alarms) + (correct neg + misses)(correct neg + false alarms))/N
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        if ffof is None:
            ffof = self.ffof

        n = ftot + ffot + ftof + ffof
        expect_corr = ((ftot + ffot)*(ftot + ftof) + (ffof + ftof)*(ffof + ffot)) / n

        return ((ftot + ffof) - expect_corr) / (n - expect_corr + 1e-5)

    def accu(self, ftot=None, ffof=None, ftof=None, ffot=None):
        '''Accuracy (fraction correct)

        Accuracy = (hits + correct_neg)/(total)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        if ffof is None:
            ffof = self.ffof

        return (ftot + ffof) / (ftot + ftof + ffot + ffof + 1e-5)

    def POD(self, ftot=None, ffot=None):
        '''Probability of detection

        POD = hits/(hits + misses) = ftot / (ftot + ffot)
        '''
        if ftot is None:
            ftot = self.ftot
        if ffot is None:
            ffot = self.ffot

        return ftot / (ftot + ffot + 1e-5)

    def FAR(self, ftof=None, ftot=None):
        '''False alarm ratio

        FAR = false alarms / (hits + false alarms)
            = ftof / (ftot + ftof)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof

        return ftof / (ftot + ftof + 1e-5)

    def BIAS(self, ftof=None, ftot=None, ffot=None):
        '''Bias score

        BIAS =  (hits + false alarms) / (hits + misses)
            = (ftot + ftof) / (ftot + ftof)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof
        if ffot is None:
            ffot = self.ffot
        return (ftot + ftof) / (ftot + ffot + 0.0001)

    def POFD(self, ftof=None, ffof=None):
        '''Probability of false detection

        POFD = false alarms / (correct neg + false alarms)
             = ftof / (ffof + ftof)
        '''
        if ftof is None:
            ftof = self.ftof
        if ffof is None:
            ffof = self.ffof

        return ftof / (ftof + ffof + 1e-5)

    def SR(self, ftot=None, ftof=None):
        '''Success ratio

        SR = hits / (hits + false alarms)
           = ftot / (ftot + ftof)
        '''
        if ftot is None:
            ftot = self.ftot
        if ftof is None:
            ftof = self.ftof

        return ftot / (ftot + ftof + 1e-5)


if __name__ == '__main__':
    # vf = verify(obs=target, pred=pred, thre_min=0.5)
    vf = verify_(hits=82, misses=23, false=38, corrneg=222, mark_key="10mm")
    print(vf.describe())
