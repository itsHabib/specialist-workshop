"""Versioned task packages: data/contract changes create new immutable versions."""
import json
import re
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from workshop.domain import digest
from workshop import store


class Case(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id: str=Field(pattern=r'^[a-zA-Z0-9_-]{1,100}$')
    family: str=Field(min_length=1,max_length=100)
    input: str | dict[str,Any]
    expected: dict[str,Any]
    provenance: str=Field(min_length=1,max_length=1000)


class Package(BaseModel):
    model_config=ConfigDict(extra='forbid')
    version: Literal[1]=1
    id: str=Field(pattern=r'^[a-z][a-z0-9-]{1,48}$')
    name: str=Field(min_length=1,max_length=100)
    instructions: str=Field(min_length=1,max_length=12000)
    output_schema: dict[str,Any]
    grader: Literal['json-exact-v1','roxiq-plan-v1']='json-exact-v1'
    unordered_fields: list[str]=Field(default_factory=list,max_length=10)
    abstain_field: str | None=None
    abstain_values: list[str]=Field(default_factory=list,max_length=12)
    train: list[Case]=Field(min_length=4,max_length=2000)
    development: list[Case]=Field(min_length=2,max_length=500)
    final: list[Case]=Field(min_length=2,max_length=500)
    provenance: dict[str,str]
    environment: Literal['ci-evidence-v1'] | None=None
    episodes: list[dict[str,Any]]=Field(default_factory=list,max_length=100)

    @model_validator(mode='after')
    def validate_contract(self):
        def reject_refs(value):
            if isinstance(value,dict):
                if any(k in value for k in ('$ref','$dynamicRef')):
                    raise ValueError('Schemas must be self-contained, with no reference resolution')
                for child in value.values():reject_refs(child)
            if isinstance(value,list):
                for child in value:reject_refs(child)
        reject_refs(self.output_schema)
        try: Draft202012Validator.check_schema(self.output_schema)
        except SchemaError as exc: raise ValueError(str(exc)[:500]) from exc
        schema=Draft202012Validator(self.output_schema)
        ids=set();families={};inputs={}
        for split in ('train','development','final'):
            for case in getattr(self,split):
                if case.id in ids:raise ValueError('Duplicate example ID')
                ids.add(case.id)
                key=input_hash(case.input)
                if key in inputs and inputs[key]!=split:raise ValueError('Input overlaps data splits')
                if case.family in families and families[case.family]!=split:raise ValueError('Family overlaps data splits')
                inputs[key]=split;families[case.family]=split
                try: schema.validate(case.expected)
                except ValidationError as exc: raise ValueError(str(exc)[:500]) from exc
                if self.grader=='roxiq-plan-v1':
                    from workshop.analytics import validate_plan
                    if not isinstance(case.input,dict) or 'context' not in case.input:raise ValueError('RoxIQ input needs context')
                    validate_plan(json.dumps(case.expected),case.input['context'])
        if self.episodes and not self.environment:raise ValueError('Episodes need an installed environment adapter')
        for episode in self.episodes:
            if episode.get('family') in {c.family for c in self.train+self.development}:
                raise ValueError('Final episode family overlaps practice/development')
        return self


def input_hash(value):
    def normalize(item):
        if isinstance(item,str):return ' '.join(item.split()).casefold()
        if isinstance(item,dict):return {k:normalize(v) for k,v in item.items()}
        if isinstance(item,list):return [normalize(v) for v in item]
        return item
    return digest(normalize(value))


def messages(package,input_value,output=None):
    content=input_value if isinstance(input_value,str) else json.dumps(input_value,separators=(',',':'))
    rows=[dict(role='system',content=package.instructions),dict(role='user',content=content)]
    if output is not None:rows.append(dict(role='assistant',content=json.dumps(output,separators=(',',':'))))
    return rows


def contract_hash(package):
    return digest(dict(instructions=package.instructions,schema=package.output_schema,grader=package.grader,
                       unordered_fields=package.unordered_fields,environment=package.environment))


def summary(package):
    fingerprint=digest(package.model_dump())
    return dict(reference=f'{package.id}@{fingerprint}',id=package.id,name=package.name,
                counts={s:len(getattr(package,s)) for s in ('train','development','final')},
                contract_hash=contract_hash(package),final_hash=digest([r.model_dump() for r in package.final]),
                provenance=package.provenance,environment=package.environment)


def import_package(package):
    info=summary(package)
    path=store.STATE/'packages'/package.id/(info['reference'].split('@')[1]+'.json')
    if path.exists():
        if store.read_json(path)!=package.model_dump():raise ValueError('Existing immutable version differs')
        return info
    store.atomic_json(path,package.model_dump())
    return info


def load(reference):
    match=re.fullmatch(r'([a-z][a-z0-9-]{1,48})@([a-f0-9]{64})',reference)
    if not match:raise ValueError('Use the exact package reference returned on import')
    path=store.STATE/'packages'/match[1]/f'{match[2]}.json'
    package=Package.model_validate(store.read_json(path))
    if digest(package.model_dump())!=match[2]:raise ValueError('Package hash changed')
    return package


def correct(reference,case):
    package=load(reference)
    if case.family in {r.family for r in package.development+package.final}:raise ValueError('Evaluation family is reserved')
    if input_hash(case.input) in {input_hash(r.input) for r in package.development+package.final}:raise ValueError('Evaluation input is reserved')
    payload=package.model_dump()
    payload['train']=[r.model_dump() for r in package.train if input_hash(r.input)!=input_hash(case.input) and r.id!=case.id]+[case.model_dump()]
    payload['provenance']['parent_version']=reference
    return import_package(Package.model_validate(payload))


def score(package,input_value,raw,expected=None):
    try:
        output=json.loads(raw)
        Draft202012Validator(package.output_schema).validate(output)
        if package.grader=='roxiq-plan-v1':
            from workshop.analytics import validate_plan,canonical
            output=validate_plan(raw,input_value['context']).model_dump()
            normalize=canonical
        else:
            def normalize(value):
                result=dict(value)
                for key in package.unordered_fields:
                    if key in result:result[key]=sorted(result[key],key=lambda v:json.dumps(v,sort_keys=True))
                return result
        correct=normalize(output)==normalize(expected) if expected is not None else None
        abstained=bool(package.abstain_field and output.get(package.abstain_field) in package.abstain_values)
        return dict(valid=True,correct=correct,accepted=correct is True,abstained=abstained,output=output,error=None)
    except Exception as exc:
        return dict(valid=False,correct=False if expected is not None else None,accepted=False,abstained=False,output=None,error=str(exc)[:600])
