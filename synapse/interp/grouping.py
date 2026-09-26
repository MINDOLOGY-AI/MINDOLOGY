# groups labeled units (SAE features, neurons) by meaning with an llm tool loop (see synapse/interp/DOC.md, feature groups).
# features are processed in batches; per batch the llm sees every existing group and the batch's labels, creates new
# groups with create_group and puts every feature into a group with assign. model-agnostic: needs only ids + labels.

import json
from collections import Counter

from synapse.interp.openrouter import chat

BATCH_SIZE = 20
MAX_TURNS = 6  # llm turns per batch before the batch must be fully assigned
MAX_TOKENS = 4000

SYSTEM = (
    "You are organizing the features of a sparse autoencoder trained on a language model into groups by meaning. "
    "Each feature has a short label describing what it activates on. You get the current list of groups and a batch "
    "of new features. Put every feature of the batch into exactly one group: reuse an existing group when the feature "
    "is about exactly that concept, otherwise create a new group with create_group (you can create several groups at "
    "once), then call assign.\n\n"
    "Groups must be SPECIFIC: one concrete concept per group. 'lizards' is a group, 'animals' is not, unless the "
    "features are literally about the general category itself (e.g. the word 'animal' or text about animal taxonomy). "
    "'Python function definitions' and 'Python import statements' are separate groups; 'code' is far too broad. Keep "
    "surface-form groups (a specific token or substring, e.g. 'the CSS value nowrap') separate from meaning groups. "
    "Creating many groups is expected and good, and a group with a single feature is fine. Never stretch an existing "
    "group to fit a feature that is only loosely related.\n\n"
    "Group name: a few words. Group description: one line saying exactly what member features activate on."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "create_group",
        "description": "Create a new group and get its id. Call it several times in one turn to create several groups.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "a few words, specific, e.g. 'lizards'"},
            "description": {"type": "string", "description": "one line: exactly what member features activate on"},
        }, "required": ["name", "description"]},
    }},
    {"type": "function", "function": {
        "name": "assign",
        "description": "Put features of the current batch into an existing group.",
        "parameters": {"type": "object", "properties": {
            "group_id": {"type": "string", "description": "id of an existing group, e.g. 'g17'"},
            "feature_ids": {"type": "array", "items": {"type": "string"}, "description": "feature ids from the batch"},
        }, "required": ["group_id", "feature_ids"]},
    }},
]


async def build_groups(features, model, log_path, batch_size=BATCH_SIZE):
    # features: [{"id": "L8:123", "label": str}] in processing order.
    # returns groups {gid: {"name": str, "desc": str}}, assign {feature id: gid}; writes one log line per batch
    groups = {}  # {gid: {"name": str, "desc": str}}
    assign = {}  # {feature id: gid}
    with open(log_path, "w") as log:
        for b in range(0, len(features), batch_size):
            batch = features[b:b + batch_size]
            ids = {f["id"] for f in batch}  # {str}
            sizes = Counter(assign.values())  # {gid: members so far}
            group_list = "\n".join(f"{g}: {v['name']} — {v['desc']} ({sizes[g]} members)" for g, v in groups.items())
            batch_list = "\n".join(f"{f['id']}: {f['label']}" for f in batch)
            messages = [  # [{"role": ..., ...}] the whole tool conversation for this batch
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Existing groups:\n{group_list or '(none yet)'}\n\nFeatures to assign:\n{batch_list}"},
            ]
            for _ in range(MAX_TURNS):
                msg = await chat(messages, model, MAX_TOKENS, tools=TOOLS)
                calls = msg.get("tool_calls") or []
                messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
                for c in calls:
                    # run the call against groups / assign; bad arguments go back to the llm as an error to fix next turn
                    name = c["function"]["name"]
                    try:
                        args = json.loads(c["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = None
                    if args is None:
                        result = "error: arguments are not valid JSON"
                    elif name == "create_group":
                        same = [g for g, v in groups.items() if v["name"].strip().lower() == str(args.get("name", "")).strip().lower()]
                        if not args.get("name") or not args.get("description"):
                            result = "error: name and description are required"
                        elif same:
                            result = f"error: group {same[0]} already has this name; assign to it or pick a more specific name"
                        else:
                            gid = f"g{len(groups)}"
                            groups[gid] = {"name": args["name"].strip(), "desc": args["description"].strip()}
                            result = f"created {gid}"
                    elif name == "assign":
                        gid = args.get("group_id")
                        fids = args.get("feature_ids") or []
                        bad = [f for f in fids if f not in ids]
                        if gid not in groups:
                            result = f"error: no group {gid}; create it first"
                        else:
                            assign.update({f: gid for f in fids if f in ids})
                            result = f"assigned {len(fids) - len(bad)} to {gid}" + (f"; error: not in this batch: {bad}" if bad else "")
                    else:
                        result = f"error: unknown tool {name}"
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                missing = sorted(ids - assign.keys())
                if not missing:
                    break
                if not calls:
                    messages.append({"role": "user", "content": f"Not assigned yet: {', '.join(missing)}. Assign every feature."})
            missing = sorted(ids - assign.keys())
            assert not missing, f"batch {b // batch_size}: unassigned after {MAX_TURNS} turns: {missing}"
            log.write(json.dumps({"batch": b // batch_size, "messages": messages}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"  batch {b // batch_size + 1}/{-(-len(features) // batch_size)}: {len(groups)} groups", flush=True)
    return groups, assign

