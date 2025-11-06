"""
Common intervention strategies for epidemic control.
"""


def lockdown(start_time: float, duration: float, effectiveness: float = 0.7):
    """
    Simulate a lockdown intervention.

    Args:
        start_time: When the lockdown begins
        duration: How long the lockdown lasts
        effectiveness: Reduction in transmission (0-1), default 0.7 = 70% reduction

    Returns:
        Dictionary with intervention parameters
    """
    return {
        'start': start_time,
        'end': start_time + duration,
        'effectiveness': effectiveness,
        'type': 'reduction',
        'name': 'Lockdown'
    }


def social_distancing(start_time: float, duration: float, effectiveness: float = 0.4):
    """
    Simulate social distancing measures.

    Args:
        start_time: When social distancing begins
        duration: How long it lasts
        effectiveness: Reduction in transmission (0-1), default 0.4 = 40% reduction

    Returns:
        Dictionary with intervention parameters
    """
    return {
        'start': start_time,
        'end': start_time + duration,
        'effectiveness': effectiveness,
        'type': 'reduction',
        'name': 'Social Distancing'
    }


def vaccination_campaign(start_time: float, duration: float, coverage: float = 0.6):
    """
    Simulate a vaccination campaign (simplified as transmission reduction).

    Args:
        start_time: When vaccination starts
        duration: Campaign duration
        coverage: Fraction of population vaccinated (0-1), default 0.6 = 60%

    Returns:
        Dictionary with intervention parameters
    """
    # Simplified: vaccination reduces effective transmission
    effectiveness = coverage * 0.85  # Assuming 85% vaccine efficacy

    return {
        'start': start_time,
        'end': start_time + duration,
        'effectiveness': effectiveness,
        'type': 'reduction',
        'name': 'Vaccination Campaign'
    }


def contact_tracing(start_time: float, duration: float, effectiveness: float = 0.3):
    """
    Simulate contact tracing intervention.

    Args:
        start_time: When contact tracing begins
        duration: How long it operates
        effectiveness: Reduction in transmission (0-1), default 0.3 = 30% reduction

    Returns:
        Dictionary with intervention parameters
    """
    return {
        'start': start_time,
        'end': start_time + duration,
        'effectiveness': effectiveness,
        'type': 'reduction',
        'name': 'Contact Tracing'
    }
