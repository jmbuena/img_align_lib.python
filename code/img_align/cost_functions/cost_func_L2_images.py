
# @brief Cost Function interface
# @author Jose M. Buenaposada
# @date 2017/08/16
#
# Grupo de investigaci'on en Percepci'on Computacional y Rob'otica)
# (Perception for Computers & Robots research Group)
# Facultad de Inform'atica (Computer Science School)
# Universidad Polit'ecnica de Madrid (UPM) (Madrid Technical University)
# http://www.dia.fi.upm.es/~pcr

import abc

class CostFunL2Images:
    """
    The interface for the Cost Functions that compare grey levels (or descriptors) of
    the template image and the input image. This cost functions departs from an
    initial guess for motion params.

    Examples of this kind of cost functions are the ones based in the Bright Constancy
    Asumption (BCA):

        I(f(x,p_t),t) = T(x)

    where T(x) is the template (object model), f(x,p_t) is the motion model and
    I(f(x,p_t), t) is the warped image (indexed with template coordinates x) taken
    from the image I(y,t).

    Different image alignment solutions arises from different asumptions based on the
    BCA:

    1)  || I(f(x, p_t+1 + \delta p), t+1) - T(x) ||^2 (Lucas-Kanade, Hager's Jacobian factorization).
    2)  || I(f(f^{-1}(x, \delta p'), p_t), t+1) - T(x) ||^2 (Baker's Inverse compositional).

    The optimizer for this kind of Cost Functions are always based in Gauss-Newton (with
    efficient solutions in some cases).
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, object_model, motion_model, show_debug_info=False):
        """

        :param object_model:
        :param motion_model:
        :return:
        """
        self.object_model = object_model
        self.motion_model = motion_model
        self.show_debug_info = show_debug_info
        return

    @abc.abstractmethod
    def computeValue(self,
                     residuals,
                     motion_params,
                     delta_params,
                     image):
        """
        :param residuals:
        :param motion_params:
        :param delta_params:
        :param image:
        :return:
        """
        return

    @abc.abstractmethod
    def computeResiduals(self, image, motion_params):
        """
        :param image:
        :param motion_params:
        :return:
        """
        return

    @abc.abstractmethod
    def computeJacobian(self, motion_params):
        """
        Computes the residuals Jacobian w.r.t. motion params.

        :param image:
        :param motion_params:
        :return:
        """
        return

    @abc.abstractmethod
    def computeJacobianPseudoInverse(self, motion_params):
        """
        Computes the residuals Jacobian pseudoinverse to solve for
        motion parameters update (inc_params). Note that usually is better to
        use SVD for solving for inc_params in:

           J x inc_params = residuals

        but with Inverse Compositional and Hager's Jacobian Factorization
        we get a very efficient way of computing the Jacobian Moore-Penrose
        pseudoinverse and get the inc_params values that way.

        :param image:
        :param motion_params:
        :return:
        """
        return

    @abc.abstractmethod
    def updateMotionParams(self, motion_params, inc_params):
        """

        :param image:
        :param motion_params:
        :return:
        """
        return


