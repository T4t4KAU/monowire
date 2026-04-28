#ifdef NDEBUG
#undef NDEBUG
#endif

#include "monowire.h"

// TODO: should not include monowire sources
#include "../src/monowire-grammar.h"

#include <cassert>

static const char * type_str(monowire_gretype type) {
    switch (type) {
        case MONOWIRE_GRETYPE_CHAR: return "MONOWIRE_GRETYPE_CHAR";
        case MONOWIRE_GRETYPE_CHAR_NOT: return "MONOWIRE_GRETYPE_CHAR_NOT";
        case MONOWIRE_GRETYPE_CHAR_ALT: return "MONOWIRE_GRETYPE_CHAR_ALT";
        case MONOWIRE_GRETYPE_CHAR_RNG_UPPER: return "MONOWIRE_GRETYPE_CHAR_RNG_UPPER";
        case MONOWIRE_GRETYPE_RULE_REF: return "MONOWIRE_GRETYPE_RULE_REF";
        case MONOWIRE_GRETYPE_ALT: return "MONOWIRE_GRETYPE_ALT";
        case MONOWIRE_GRETYPE_END: return "MONOWIRE_GRETYPE_END";
        default: return "?";
    }
}

static void verify_parsing(const char *grammar_bytes, const std::vector<std::pair<std::string, uint32_t>> expected, const std::vector<monowire_grammar_element> &expected_rules) {
    uint32_t index = 0;
    monowire_grammar_parser parsed_grammar;
    parsed_grammar.parse(grammar_bytes);

    std::map<uint32_t, std::string> symbol_names;
    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
        symbol_names[it->second] = it->first;
    }

    auto print_all = [&]() {
        fprintf(stderr, "    verify_parsing(R\"\"\"(%s)\"\"\", {\n", grammar_bytes);
        for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it) {
            fprintf(stderr, "        {\"%s\", %u},\n", it->first.c_str(), it->second);
        }
        fprintf(stderr, "    }, {\n");
        for (size_t i_rule = 0; i_rule < parsed_grammar.rules.size(); i_rule++) {
            fprintf(stderr, "        // %s (index %zu)\n", symbol_names[i_rule].c_str(), i_rule);
            auto & rule = parsed_grammar.rules[i_rule];
            for (uint32_t i = 0; i < rule.size(); i++) {
                std::string rule_str;
                fprintf(stderr, "        {%s, ", type_str(rule[i].type));
                if (rule[i].type == MONOWIRE_GRETYPE_CHAR || rule[i].type == MONOWIRE_GRETYPE_CHAR_ALT ||
                    rule[i].type == MONOWIRE_GRETYPE_CHAR_NOT || rule[i].type == MONOWIRE_GRETYPE_CHAR_RNG_UPPER) {
                    char c = rule[i].value;
                    if (c == '\n') {
                        fprintf(stderr, "'\\n'");
                    } else if (c == '\t') {
                        fprintf(stderr, "'\\t'");
                    } else if (c == '\r') {
                        fprintf(stderr, "'\\r'");
                    } else if (c == '\0') {
                        fprintf(stderr, "'\\0'");
                    } else {
                        fprintf(stderr, "'%c'", c);
                    }
                } else if (rule[i].type == MONOWIRE_GRETYPE_RULE_REF) {
                    fprintf(stderr, "/* %s */ %u", symbol_names[rule[i].value].c_str(), rule[i].value);
                } else {
                    fprintf(stderr, "%u", rule[i].value);
                }
                fprintf(stderr, "},\n");
            }
        }
        fprintf(stderr, "    });\n");
    };

    if (getenv("TEST_GRAMMAR_PARSER_PRINT_ALL")) {
        print_all();
        fprintf(stderr, "\n");
        return;
    }

    fprintf(stderr, "Testing grammar:%s\n", grammar_bytes);

    if (parsed_grammar.symbol_ids.size() != expected.size()) {
        fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
        print_all();
        assert(parsed_grammar.symbol_ids.size() == expected.size());
    }

    for (auto it = parsed_grammar.symbol_ids.begin(); it != parsed_grammar.symbol_ids.end(); ++it)
    {
        std::string key = it->first;
        uint32_t value = it->second;
        std::pair<std::string, uint32_t> expected_pair = expected[index];

        // pretty print error message before asserting
        if (expected_pair.first != key || expected_pair.second != value)
        {
            fprintf(stderr, "index: %u\n", index);
            fprintf(stderr, "expected_pair: %s, %u\n", expected_pair.first.c_str(), expected_pair.second);
            fprintf(stderr, "actual_pair: %s, %u\n", key.c_str(), value);
            fprintf(stderr, "expected_pair != actual_pair\n");
            fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
            print_all();
        }

        assert(expected_pair.first == key && expected_pair.second == value);

        index++;
    }

    index = 0;
    for (auto rule : parsed_grammar.rules)
    {
        // compare rule to expected rule
        for (uint32_t i = 0; i < rule.size(); i++)
        {
            monowire_grammar_element element = rule[i];
            monowire_grammar_element expected_element = expected_rules[index];

            // pretty print error message before asserting
            if (expected_element.type != element.type || expected_element.value != element.value)
            {
                fprintf(stderr, "index: %u\n", index);
                fprintf(stderr, "expected_element: %s, %u\n", type_str(expected_element.type), expected_element.value);
                fprintf(stderr, "actual_element: %s, %u\n", type_str(element.type), element.value);
                fprintf(stderr, "expected_element != actual_element\n");
                fprintf(stderr, "all elements:\n");
                fprintf(stderr, "Code to update expectation (set TEST_GRAMMAR_PARSER_PRINT_ALL=1 to print all):\n");
                print_all();
            }

            assert(expected_element.type == element.type && expected_element.value == element.value);
            index++;
        }
    }
}

static void verify_failure(const char * grammar_bytes) {
    fprintf(stderr, "Testing expected failure:%s\n", grammar_bytes);
    monowire_grammar_parser result;
    result.parse(grammar_bytes);
    assert(result.rules.empty() && "should have failed");
}

int main()
{
    verify_failure(R"""(
        root ::= "a"{,}"
    )""");

    verify_failure(R"""(
        root ::= (((((([^x]*){0,99}){0,99}){0,99}){0,99}){0,99}){0,99}
    )""");

    verify_failure(R"""(
        root ::= "a"{,10}"
    )""");

    verify_parsing(R"""(
        root  ::= "a"
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a" | [bdx-z] | [^1-3]
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_CHAR, 'b'},
        {MONOWIRE_GRETYPE_CHAR_ALT, 'd'},
        {MONOWIRE_GRETYPE_CHAR_ALT, 'x'},
        {MONOWIRE_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_CHAR_NOT, '1'},
        {MONOWIRE_GRETYPE_CHAR_RNG_UPPER, '3'},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= a+
        a     ::= "a"
    )""", {
        {"a", 1},
        {"root", 0},
        {"root_2", 2},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_RULE_REF, /* a */ 1},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_2 */ 2},
        {MONOWIRE_GRETYPE_END, 0},
        // a (index 1)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_END, 0},
        // root_2 (index 2)
        {MONOWIRE_GRETYPE_RULE_REF, /* a */ 1},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_2 */ 2},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"+
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_END, 0},
        // root_1 (index 1)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= a?
        a     ::= "a"
    )""", {
        {"a", 1},
        {"root", 0},
        {"root_2", 2},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_RULE_REF, /* root_2 */ 2},
        {MONOWIRE_GRETYPE_END, 0},
        // a (index 1)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_END, 0},
        // root_2 (index 2)
        {MONOWIRE_GRETYPE_RULE_REF, /* a */ 1},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"?
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_END, 0},
        // root_1 (index 1)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= a*
        a     ::= "a"
    )""", {
        {"a", 1},
        {"root", 0},
        {"root_2", 2},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_RULE_REF, /* root_2 */ 2},
        {MONOWIRE_GRETYPE_END, 0},
        // a (index 1)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_END, 0},
        // root_2 (index 2)
        {MONOWIRE_GRETYPE_RULE_REF, /* a */ 1},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_2 */ 2},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"*
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_END, 0},
        // root_1 (index 1)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2}
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2,}
    )""", {
        {"root", 0},
        {"root_1", 1},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_END, 0},
        // root_1 (index 1)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{ 4}
    )""", {
        {"root", 0},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= "a"{2,4}
    )""", {
        {"root", 0},
        {"root_1", 1},
        {"root_2", 2},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_2 */ 2},
        {MONOWIRE_GRETYPE_END, 0},
        // root_1 (index 1)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
        // root_2 (index 2)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= (expr "=" term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= [0-9]+
    )""", {
        {"expr", 2},
        {"expr_5", 5},
        {"expr_6", 6},
        {"root", 0},
        {"root_1", 1},
        {"root_4", 4},
        {"term", 3},
        {"term_7", 7},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_4 */ 4},
        {MONOWIRE_GRETYPE_END, 0},
        // root_1 (index 1)
        {MONOWIRE_GRETYPE_RULE_REF, /* expr */ 2},
        {MONOWIRE_GRETYPE_CHAR, '='},
        {MONOWIRE_GRETYPE_RULE_REF, /* term */ 3},
        {MONOWIRE_GRETYPE_CHAR, '\n'},
        {MONOWIRE_GRETYPE_END, 0},
        // expr (index 2)
        {MONOWIRE_GRETYPE_RULE_REF, /* term */ 3},
        {MONOWIRE_GRETYPE_RULE_REF, /* expr_6 */ 6},
        {MONOWIRE_GRETYPE_END, 0},
        // term (index 3)
        {MONOWIRE_GRETYPE_CHAR, '0'},
        {MONOWIRE_GRETYPE_CHAR_RNG_UPPER, '9'},
        {MONOWIRE_GRETYPE_RULE_REF, /* term_7 */ 7},
        {MONOWIRE_GRETYPE_END, 0},
        // root_4 (index 4)
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_4 */ 4},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
        // expr_5 (index 5)
        {MONOWIRE_GRETYPE_CHAR, '-'},
        {MONOWIRE_GRETYPE_CHAR_ALT, '+'},
        {MONOWIRE_GRETYPE_CHAR_ALT, '*'},
        {MONOWIRE_GRETYPE_CHAR_ALT, '/'},
        {MONOWIRE_GRETYPE_RULE_REF, /* term */ 3},
        {MONOWIRE_GRETYPE_END, 0},
        // expr_6 (index 6)
        {MONOWIRE_GRETYPE_RULE_REF, /* expr_5 */ 5},
        {MONOWIRE_GRETYPE_RULE_REF, /* expr_6 */ 6},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
        // term_7 (index 7)
        {MONOWIRE_GRETYPE_CHAR, '0'},
        {MONOWIRE_GRETYPE_CHAR_RNG_UPPER, '9'},
        {MONOWIRE_GRETYPE_RULE_REF, /* term_7 */ 7},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    verify_parsing(R"""(
        root  ::= (expr "=" ws term "\n")+
        expr  ::= term ([-+*/] term)*
        term  ::= ident | num | "(" ws expr ")" ws
        ident ::= [a-z] [a-z0-9_]* ws
        num   ::= [0-9]+ ws
        ws    ::= [ \t\n]*
    )""", {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"ident", 8},
        {"ident_10", 10},
        {"num", 9},
        {"num_11", 11},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"term", 4},
        {"ws", 3},
        {"ws_12", 12},
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_5 */ 5},
        {MONOWIRE_GRETYPE_END, 0},
        // root_1 (index 1)
        {MONOWIRE_GRETYPE_RULE_REF, /* expr */ 2},
        {MONOWIRE_GRETYPE_CHAR, '='},
        {MONOWIRE_GRETYPE_RULE_REF, /* ws */ 3},
        {MONOWIRE_GRETYPE_RULE_REF, /* term */ 4},
        {MONOWIRE_GRETYPE_CHAR, '\n'},
        {MONOWIRE_GRETYPE_END, 0},
        // expr (index 2)
        {MONOWIRE_GRETYPE_RULE_REF, /* term */ 4},
        {MONOWIRE_GRETYPE_RULE_REF, /* expr_7 */ 7},
        {MONOWIRE_GRETYPE_END, 0},
        // ws (index 3)
        {MONOWIRE_GRETYPE_RULE_REF, /* ws_12 */ 12},
        {MONOWIRE_GRETYPE_END, 0},
        // term (index 4)
        {MONOWIRE_GRETYPE_RULE_REF, /* ident */ 8},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_RULE_REF, /* num */ 9},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_CHAR, '('},
        {MONOWIRE_GRETYPE_RULE_REF, /* ws */ 3},
        {MONOWIRE_GRETYPE_RULE_REF, /* expr */ 2},
        {MONOWIRE_GRETYPE_CHAR, ')'},
        {MONOWIRE_GRETYPE_RULE_REF, /* ws */ 3},
        {MONOWIRE_GRETYPE_END, 0},
        // root_5 (index 5)
        {MONOWIRE_GRETYPE_RULE_REF, /* root_1 */ 1},
        {MONOWIRE_GRETYPE_RULE_REF, /* root_5 */ 5},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
        // expr_6 (index 6)
        {MONOWIRE_GRETYPE_CHAR, '-'},
        {MONOWIRE_GRETYPE_CHAR_ALT, '+'},
        {MONOWIRE_GRETYPE_CHAR_ALT, '*'},
        {MONOWIRE_GRETYPE_CHAR_ALT, '/'},
        {MONOWIRE_GRETYPE_RULE_REF, /* term */ 4},
        {MONOWIRE_GRETYPE_END, 0},
        // expr_7 (index 7)
        {MONOWIRE_GRETYPE_RULE_REF, /* expr_6 */ 6},
        {MONOWIRE_GRETYPE_RULE_REF, /* expr_7 */ 7},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
        // ident (index 8)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {MONOWIRE_GRETYPE_RULE_REF, /* ident_10 */ 10},
        {MONOWIRE_GRETYPE_RULE_REF, /* ws */ 3},
        {MONOWIRE_GRETYPE_END, 0},
        // num (index 9)
        {MONOWIRE_GRETYPE_CHAR, '0'},
        {MONOWIRE_GRETYPE_CHAR_RNG_UPPER, '9'},
        {MONOWIRE_GRETYPE_RULE_REF, /* num_11 */ 11},
        {MONOWIRE_GRETYPE_RULE_REF, /* ws */ 3},
        {MONOWIRE_GRETYPE_END, 0},
        // ident_10 (index 10)
        {MONOWIRE_GRETYPE_CHAR, 'a'},
        {MONOWIRE_GRETYPE_CHAR_RNG_UPPER, 'z'},
        {MONOWIRE_GRETYPE_CHAR_ALT, '0'},
        {MONOWIRE_GRETYPE_CHAR_RNG_UPPER, '9'},
        {MONOWIRE_GRETYPE_CHAR_ALT, '_'},
        {MONOWIRE_GRETYPE_RULE_REF, /* ident_10 */ 10},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
        // num_11 (index 11)
        {MONOWIRE_GRETYPE_CHAR, '0'},
        {MONOWIRE_GRETYPE_CHAR_RNG_UPPER, '9'},
        {MONOWIRE_GRETYPE_RULE_REF, /* num_11 */ 11},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
        // ws_12 (index 12)
        {MONOWIRE_GRETYPE_CHAR, ' '},
        {MONOWIRE_GRETYPE_CHAR_ALT, '\t'},
        {MONOWIRE_GRETYPE_CHAR_ALT, '\n'},
        {MONOWIRE_GRETYPE_RULE_REF, /* ws_12 */ 12},
        {MONOWIRE_GRETYPE_ALT, 0},
        {MONOWIRE_GRETYPE_END, 0},
    });

    // <[1000]> = "<think>"
    // <[1001]> = "</think>"
    verify_parsing(R"""(
        root  ::= <[1000]> !<[1001]> <[1001]>
    )""", {
        {"root", 0}
    }, {
        // root (index 0)
        {MONOWIRE_GRETYPE_TOKEN, 1000},
        {MONOWIRE_GRETYPE_TOKEN_NOT, 1001},
        {MONOWIRE_GRETYPE_TOKEN, 1001},
        {MONOWIRE_GRETYPE_END, 0},
    });

    return 0;
}
