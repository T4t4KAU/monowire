#pragma once

#include "monowire.h"

#include <map>
#include <regex>
#include <string>
#include <vector>

struct monowire_vocab;

// grammar element type
enum monowire_gretype {
    // end of rule definition
    MONOWIRE_GRETYPE_END = 0,

    // start of alternate definition for rule
    MONOWIRE_GRETYPE_ALT = 1,

    // non-terminal element: reference to rule
    MONOWIRE_GRETYPE_RULE_REF = 2,

    // terminal element: character (code point)
    MONOWIRE_GRETYPE_CHAR = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    MONOWIRE_GRETYPE_CHAR_NOT = 4,

    // modifies a preceding MONOWIRE_GRETYPE_CHAR or MONOWIRE_GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    MONOWIRE_GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding MONOWIRE_GRETYPE_CHAR or
    // MONOWIRE_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab],
    // [a-zA])
    MONOWIRE_GRETYPE_CHAR_ALT = 6,

    // any character (.)
    MONOWIRE_GRETYPE_CHAR_ANY = 7,

    // terminal element: token (<[token-id]>)
    MONOWIRE_GRETYPE_TOKEN = 8,

    // inverse token (!<[token-id]>)
    MONOWIRE_GRETYPE_TOKEN_NOT = 9,
};

typedef struct monowire_grammar_element {
    enum monowire_gretype type;
    uint32_t value; // Unicode code point, rule ID, or token ID
} monowire_grammar_element;

struct monowire_partial_utf8 {
    uint32_t value; // bit value so far (unshifted)
    int n_remain;   // num bytes remaining; -1 indicates invalid sequence
};

struct monowire_grammar_candidate {
    size_t index;
    const uint32_t *code_points;
    monowire_partial_utf8 partial_utf8;
    monowire_token id;
};

using monowire_grammar_rule = std::vector<monowire_grammar_element>;
using monowire_grammar_stack = std::vector<const monowire_grammar_element *>;

using monowire_grammar_rules = std::vector<monowire_grammar_rule>;
using monowire_grammar_stacks = std::vector<monowire_grammar_stack>;
using monowire_grammar_candidates = std::vector<monowire_grammar_candidate>;

// TODO: remove, needed for tests atm
const monowire_grammar_rules &monowire_grammar_get_rules(const struct monowire_grammar *grammar);
monowire_grammar_stacks &monowire_grammar_get_stacks(struct monowire_grammar *grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `monowire_grammar_advance_stack`),
// and produces the N possible stacks if the given char is accepted at those
// positions
void monowire_grammar_accept(struct monowire_grammar *grammar, uint32_t chr);

std::vector<monowire_grammar_candidate>
monowire_grammar_reject_candidates_for_stack(const monowire_grammar_rules &rules, const monowire_grammar_stack &stack,
                                             const monowire_grammar_candidates &candidates);

struct monowire_grammar_parser {
    const monowire_vocab *vocab;
    std::map<std::string, uint32_t> symbol_ids;

    monowire_grammar_rules rules;

    monowire_grammar_parser(const struct monowire_vocab *vocab = nullptr) : vocab(vocab) {}

    monowire_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char *src, size_t len);
    uint32_t generate_symbol_id(const std::string &base_name);

    void add_rule(uint32_t rule_id, const monowire_grammar_rule &rule);

    const char *parse_alternates(const char *src, const std::string &rule_name, uint32_t rule_id, bool is_nested);

    const char *parse_sequence(const char *src, const std::string &rule_name, monowire_grammar_rule &rule,
                               bool is_nested);

    const char *parse_rule(const char *src);

    bool parse(const char *src);
    void print(FILE *file);
};

struct monowire_grammar_trigger_pattern {
    std::string pattern;
    std::regex regex;

    size_t find(const std::string &input) const;
};

struct monowire_grammar {
    // maintain a list of monowire_tokens and their positions in the
    // trigger_buffer
    using token_pos = std::pair<monowire_token, std::pair<size_t, size_t>>;

    // note: allow null vocab for testing (not great)
    const monowire_vocab *vocab;

    const monowire_grammar_rules rules; // TODO: shared ptr
    monowire_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    monowire_partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the
    // sampling. we still have trigger_tokens for non-lazy grammars to force
    // printing of special trigger tokens. (useful e.g. for tool_choice=required)
    bool lazy = false;
    bool awaiting_trigger = false;                   // Initialized to true for lazy grammars only
    std::string trigger_buffer;                      // Output buffered by lazy grammar. Will be
                                                     // cleared once trigger is found.
    std::vector<token_pos> trigger_buffer_positions; // Tokens buffered by lazy grammar. Used to
                                                     // replay when a trigger is found.
    std::vector<monowire_token> trigger_tokens;      // Tokens that trigger a lazy grammar, or tokens to force
                                                     // printing of (even if special).
    std::vector<monowire_grammar_trigger_pattern> trigger_patterns; // Regular expressions that trigger a lazy grammar.
                                                                    // Must be a full match of the entire generated
                                                                    // string, and the grammar will be given the string
                                                                    // from the first match group onwards.
};

//
// internal API
//

// note: needed for tests (not great)
struct monowire_grammar *monowire_grammar_init_impl(const struct monowire_vocab *vocab,
                                                    const monowire_grammar_element **rules, size_t n_rules,
                                                    size_t start_rule_index);

struct monowire_grammar *monowire_grammar_init_impl(const struct monowire_vocab *vocab, const char *grammar_str,
                                                    const char *grammar_root, bool lazy, const char **trigger_patterns,
                                                    size_t num_trigger_patterns, const monowire_token *trigger_tokens,
                                                    size_t num_trigger_tokens);

void monowire_grammar_free_impl(struct monowire_grammar *grammar);

struct monowire_grammar *monowire_grammar_clone_impl(const struct monowire_grammar &grammar);

// TODO: move the API below as member functions of monowire_grammar
void monowire_grammar_apply_impl(const struct monowire_grammar &grammar, monowire_token_data_array *cur_p);

void monowire_grammar_accept_impl(struct monowire_grammar &grammar, monowire_token token);

void monowire_grammar_accept_str(struct monowire_grammar &grammar, const std::string &piece);

void monowire_grammar_accept_token(struct monowire_grammar &grammar, monowire_token token, const std::string &piece);
