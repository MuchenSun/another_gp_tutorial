#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["setup_likelihood_sampler", "setup_conditional_sampler_nonzero_mean", "setup_conditional_sampler_mean"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

from ipywidgets import interact, interactive
import ipywidgets as widgets

from .data import true_parameters
from .transit_model import SillyTransitModel


def setup_gaussian_distr_1D(pdf):
    x = np.linspace(-5, 5, 500)
    def draw_pdf(mean, stddev):
        ll = pdf(x, mean, stddev)
        plt.plot(x, ll)
        plt.xlim(-5, 5)
        plt.ylim(0, 2)
        plt.show()
    w = interact(draw_pdf, mean=(-3.,3.), stddev=(0.01,3.))

def setup_gaussian_distr_2D(pdf):
    xy = np.meshgrid(*[np.linspace(-2, 2, 50) for _ in range(2)])
    grids = np.array([xy[0].ravel(), xy[1].ravel()]).T
    def draw_pdf(off_diagonal):
        mu = np.array([0., 0.])
        sig = np.array([[1., off_diagonal], [off_diagonal, 1.]])
        vals = []
        for g in grids:
            ll = pdf(g, mu, sig)
            vals.append(ll)
        vals = np.array(vals).reshape(50, 50)
        plt.contourf(xy[0], xy[1], vals, levels=40)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()
    w = interact(draw_pdf, off_diagonal=(-0.8,0.8))

def setup_gaussian_distr_2D_conditional(pdf1, pdf2):
    xy = np.meshgrid(*[np.linspace(-2, 2, 50) for _ in range(2)])
    grids = np.array([xy[0].ravel(), xy[1].ravel()]).T
    def draw_pdf(off_diagonal, y):
        mu = np.array([0., 0.])
        sig = np.array([[1., off_diagonal], [off_diagonal, 1.]])
        vals = []
        for g in grids:
            ll = pdf1(g, mu, sig)
            vals.append(ll)
        vals = np.array(vals).reshape(50, 50)

        fig = plt.figure()
        ax1 = fig.add_subplot(121, adjustable='box')
        ax1.contourf(xy[0], xy[1], vals, levels=40)
        ax1.axvline(x=y, c='r')
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_xlabel('y')
        ax1.set_ylabel('x')
        ax1.set_aspect('equal')

        x = np.linspace(-2., 2., 50)
        vals2 = pdf2(x, y, mu, sig)
        ax2 = fig.add_subplot(122)
        ax2.plot(vals2, x)
        ax2.set_xlim(0., 0.7)
        ax2.set_ylim(-2., 2.)
        asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
        ax2.set_aspect(asp)

        plt.show()
    w = interact(draw_pdf, off_diagonal=(-0.8,0.8), y=(-2.,2.))

def setup_likelihood_sampler(kernel):
    # Pre-compute some stuff.
    x0 = np.linspace(-20, 20, 100)
    r = np.abs(x0[:, None] - x0[None, :])

    # This function samples from the likelihood distribution.
    def sample_likelihood(amp, ell):
        rng = np.random.RandomState(123)
        K = kernel([amp, ell], r)
        K[np.diag_indices_from(K)] += 1e-9
        y0 = rng.multivariate_normal(np.zeros_like(x0), K, 6)
        plt.plot(x0, y0.T, "k", alpha=0.5)
        plt.ylim(-800, 800)
        plt.xlim(-20, 20)
        plt.show()

    w = interact(sample_likelihood, amp=(10, 500.0), ell=(1.0, 10.0))
    return w

def setup_conditional_sampler_zero_mean(x, y, yerr, kernel, training_func):
    # Pre-compute a bunch of things for speed.
    xs = np.linspace(-20, 20, 100)
    rxx = np.abs(x[:, None] - x[None, :])
    rss = np.abs(xs[:, None] - xs[None, :])
    rxs = np.abs(x[None, :] - xs[:, None])
    ye2 = yerr ** 2

    model = SillyTransitModel(*true_parameters)

    # This function samples from the conditional distribution and
    # plots those samples.
    def sample_conditional(amp, ell):
        rng = np.random.RandomState(123)

        # Compute the covariance matrices.
        Kxx = kernel([amp, ell], rxx)
        Kxx[np.diag_indices_from(Kxx)] += ye2
        Kss = kernel([amp, ell], rss)
        Kxs = kernel([amp, ell], rxs)

        # Compute the predictive mean, covariance, and likelihood
        mu, cov, ll = training_func(y, Kxx, Kxs, Kss)
        pred_var = np.diag(cov)

        # compute true model values
        model_ys = model.get_value(xs)

        # Sample and display the results.
        y0 = rng.multivariate_normal(mu, cov, 6)
        plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
        plt.plot(xs, mu, '--b')
        plt.fill_between(xs, mu-np.sqrt(pred_var), mu+np.sqrt(pred_var),
                            color='k', alpha=0.2)
        plt.plot(xs, y0.T, "k", alpha=0.5)
        plt.plot(xs, model_ys, ":r")
        plt.ylim(-250, 100)
        plt.xlim(-20, 20)
        plt.title("lnlike = {0}".format(ll))
        plt.show()

    w = interact(sample_conditional, amp=(10.0, 500.0), ell=(0.5, 10.0))
    return w

def setup_conditional_sampler_nonzero_mean(x, y, yerr, kernel, training_func):
    # Pre-compute a bunch of things for speed.
    xs = np.linspace(-20, 20, 100)
    rxx = np.abs(x[:, None] - x[None, :])
    rss = np.abs(xs[:, None] - xs[None, :])
    rxs = np.abs(x[None, :] - xs[:, None])
    ye2 = yerr ** 2

    model = SillyTransitModel(*true_parameters)

    # This function samples from the conditional distribution and
    # plots those samples.
    def sample_conditional(amp, ell,
                           depth=np.exp(true_parameters[0]),
                           duration=np.exp(true_parameters[1]),
                           time=true_parameters[2]):
        rng = np.random.RandomState(123)

        # Compute the covariance matrices.
        Kxx = kernel([amp, ell], rxx)
        Kxx[np.diag_indices_from(Kxx)] += ye2
        Kss = kernel([amp, ell], rss)
        Kxs = kernel([amp, ell], rxs)

        # compute true model values
        model.set_parameter_vector([np.log(depth), np.log(duration), time])
        model_y = model.get_value(x)
        model_ys = model.get_value(xs)

        # Compute the predictive mean, covariance, and likelihood
        mu, cov, ll = training_func(y, model_y, model_ys, Kxx, Kxs, Kss)
        pred_var = np.diag(cov)

        # Sample and display the results.
        y0 = rng.multivariate_normal(mu, cov, 6)
        plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
        plt.plot(xs, mu, '--b')
        plt.fill_between(xs, mu-np.sqrt(pred_var), mu+np.sqrt(pred_var),
                            color='k', alpha=0.2)
        plt.plot(xs, y0.T, "k", alpha=0.5)
        plt.plot(xs, model_ys, ":r")
        plt.ylim(-250, 100)
        plt.xlim(-20, 20)
        plt.title("lnlike = {0}".format(ll))
        plt.show()

    w = interactive(sample_conditional,
                 amp=(10.0, 500.0), ell=(0.5, 10.0),
                 depth=(10.0, 500.0),
                 duration=(0.1, 2.0),
                 time=(-5.0, 5.0))
    return w

def setup_conditional_sampler_nonzero_mean_given_param(x, y, yerr, kernel, training_func, log_depth, log_duration, time, log_amp, log_ell):
    # Pre-compute a bunch of things for speed.
    xs = np.linspace(-20, 20, 100)
    rxx = np.abs(x[:, None] - x[None, :])
    rss = np.abs(xs[:, None] - xs[None, :])
    rxs = np.abs(x[None, :] - xs[:, None])
    ye2 = yerr ** 2

    # model = SillyTransitModel(log_depth=np.log(depth), log_duration=np.log(duration), time=time)
    model = SillyTransitModel(log_depth=log_depth, log_duration=log_duration, time=time)

    # This function samples from the conditional distribution and
    # plots those samples.

    rng = np.random.RandomState(123)

    # Compute the covariance matrices.
    Kxx = kernel([np.exp(log_amp), np.exp(log_ell)], rxx)
    Kxx[np.diag_indices_from(Kxx)] += ye2
    Kss = kernel([np.exp(log_amp), np.exp(log_ell)], rss)
    Kxs = kernel([np.exp(log_amp), np.exp(log_ell)], rxs)

    # compute true model values
    # model.set_parameter_vector([np.log(depth), np.log(duration), time])
    model_y = model.get_value(x)
    model_ys = model.get_value(xs)

    # Compute the predictive mean, covariance, and likelihood
    mu, cov, ll = training_func(y, model_y, model_ys, Kxx, Kxs, Kss)
    pred_var = np.diag(cov)

    # Sample and display the results.
    y0 = rng.multivariate_normal(mu, cov, 6)
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(xs, mu, '--b')
    plt.fill_between(xs, mu-np.sqrt(pred_var), mu+np.sqrt(pred_var),
                        color='k', alpha=0.2)
    plt.plot(xs, y0.T, "k", alpha=0.5)
    plt.plot(xs, model_ys, ":r")
    plt.ylim(-250, 100)
    plt.xlim(-20, 20)
    plt.title("lnlike = {0}".format(ll))
    plt.show()

