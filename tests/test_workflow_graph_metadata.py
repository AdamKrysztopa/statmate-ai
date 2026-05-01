from statmate.api.services.workflow_graph_service import render_workflow_graph
from statmate.workflow.graph_metadata import map_step_to_node_id, workflow_payload


def test_map_step_to_node_id_normalizes_labels():
    assert map_step_to_node_id('Initialization') == 'initialization_agent'
    assert map_step_to_node_id('Reviewer Agent') == 'reviewer_agent'
    assert map_step_to_node_id('Parametric assumptions hold?') == 'parametric_assumptions_hold'


def test_workflow_payload_tracks_progress():
    steps = [{'step': 'Initialization', 'node': 'Initialization Agent', 'timestamp': 't0'}]
    payload = workflow_payload(steps, {'chosen_test': 'Independent t-test'})
    assert payload['visited_nodes'][0] == 'start'
    assert payload['active_node'] == 'initialization_agent'
    assert 'nodes' in payload and 'edges' in payload
    assert 'independent_t_test' in (payload['selected_path'] or [])


def test_render_workflow_graph_returns_svg():
    base = workflow_payload([], None)
    assets = render_workflow_graph(base)
    assert assets['svg_base64']
    assert assets['alt']


def test_workflow_graph_contains_regression_node():
    from statmate.workflow.graph_metadata import get_workflow_graph

    graph = get_workflow_graph()
    node_ids = [n['id'] for n in graph['nodes']]
    assert 'regression_node' in node_ids
