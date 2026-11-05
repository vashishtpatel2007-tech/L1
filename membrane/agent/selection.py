"""Resolve a command from surveyed camera observations, never engine truth."""
from dataclasses import dataclass, field
from .perception import Detection, Request


@dataclass
class Selection:
    target: Detection | None = None
    code: str = ''
    message: str = ''
    candidates: list[str] = field(default_factory=list)

    def issue(self):
        return {'code': self.code, 'message': self.message, 'candidates': self.candidates}


def resolve(request: Request, objects: list[Detection]) -> Selection:
    """Never use distance as an implicit tie-break for a vague command.

    Height/volume comparisons need unclipped observations and a separation
    larger than measurement tolerance. Shape uncertainty requests clarification
    instead of substituting a different shape.
    """
    if request.is_empty:
        return Selection(code='more_specific', message='Please name a colour, shape or size, for example “the red sphere” or “the tallest object”.')
    pool = [o for o in objects if not request.colour or o.colour == request.colour]
    if request.shape:
        pool = [o for o in pool if o.shape == request.shape]
    description = ' '.join(x for x in (request.colour, request.shape) if x) or 'object'
    if not pool:
        return Selection(code='not_verified', message=f'I could not verify a matching {description} in the camera scan. Please check the description or reveal the object.')

    metric, reverse = (request.superlative if request.superlative else
                       (request.size_metric, request.size == 2))
    comparing = request.superlative is not None or request.size is not None
    if comparing:
        if metric in ('height', 'volume') and any(not o.height_valid for o in pool):
            return Selection(code='measurement_uncertain', message=f'I cannot reliably compare the {description} sizes because a matching object is partly hidden. Please reveal it before retrying.')
        ordered = sorted(pool, key=lambda o: getattr(o, metric), reverse=reverse)
        if request.size == 1:
            if len(ordered) % 2 == 0:
                return Selection(code='ambiguous', message=f'There is no unique medium {description}. Please specify another feature.')
            chosen = ordered[len(ordered)//2]
        else:
            chosen = ordered[0]
        value = getattr(chosen, metric)
        # Pixel-derived measurements have noise. Close rankings need a person,
        # not an arbitrary stable-sort winner.
        floor = .002 if metric == 'volume' else .025
        tolerance = max(floor, abs(value) * .08)
        pool = [o for o in ordered if abs(getattr(o, metric)-value) <= tolerance]
    if len(pool) == 1:
        return Selection(target=pool[0])

    choices = [f'{o.colour} {o.shape}, estimated height {o.height:.2f} m' for o in pool]
    noun = description + ('es' if description.endswith('box') else 's')
    return Selection(code='ambiguous',
                     message=f'I found {len(pool)} matching {noun}. Please be more specific—for example, the taller or smaller one, or add its colour or shape.',
                     candidates=choices)
